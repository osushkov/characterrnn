
#include "RNN.hpp"
#include "../../common/Maybe.hpp"
#include "../Activations.hpp"
#include "Layer.hpp"
#include "LayerMemory.hpp"
#include <cassert>
#include <utility>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

struct ConnectionAccum {
  EMatrix accumWeightsDelta;
  unsigned samples;

  ConnectionAccum(const EMatrix &delta) : accumWeightsDelta(delta), samples(1) {}

  EMatrix GetWeightsDelta(void) const {
    assert(samples > 0);
    return accumWeightsDelta * (1.0f / samples);
  }

  void AccumDelta(const EMatrix &delta) {
    accumWeightsDelta += delta;
    samples++;
  }
};

struct NetworkWeightsAccum {
  vector<pair<LayerConnection, ConnectionAccum>> allWeightsAccum;

  void IncrementWeights(const LayerConnection &connection, const EMatrix &weightsDelta) {
    for (auto &wa : allWeightsAccum) {
      if (wa.first == connection) {
        wa.second.AccumDelta(weightsDelta);
        return;
      }
    }

    allWeightsAccum.emplace_back(connection, ConnectionAccum(weightsDelta));
  }

  Maybe<EMatrix> GetAccumDelta(const LayerConnection &connection) {
    for (auto &wa : allWeightsAccum) {
      if (wa.first == connection) {
        return Maybe<EMatrix>(wa.second.GetWeightsDelta());
      }
    }

    return Maybe<EMatrix>::none;
  }
};

struct RNN::RNNImpl {
  RNNSpec spec;
  vector<Layer> layers;
  Maybe<TimeSlice> previous;

  RNNImpl(const RNNSpec &spec) : spec(spec), previous(Maybe<TimeSlice>::none) {
    for (const auto &ls : spec.layers) {
      layers.emplace_back(spec, ls);
    }
  }

  void ClearMemory(void) { previous = Maybe<TimeSlice>::none; }

  EMatrix Process(const EMatrix &input) {
    assert(input.rows() == spec.numInputs);

    TimeSlice *prevSlice = previous.valid() ? &(previous.val()) : nullptr;
    TimeSlice curSlice(0, input, layers);

    EMatrix output = forwardPass(input, prevSlice, curSlice);
    previous = Maybe<TimeSlice>(curSlice);
    return output;
  }

  math::Tensor ComputeGradient(const vector<SliceBatch> &trace) {
    LayerMemory memory;

    vector<EMatrix> traceOutputs;
    traceOutputs.reserve(trace.size());

    // Forward pass
    TimeSlice *prevSlice = nullptr;
    for (unsigned i = 0; i < trace.size(); i++) {
      TimeSlice curSlice(i, trace[i].batchInput, layers);

      EMatrix out = forwardPass(trace[i].batchInput, prevSlice, curSlice);
      traceOutputs.push_back(out);

      prevSlice = memory.PushNewSlice(curSlice);
    }

    assert(trace.size() == traceOutputs.size());

    // Backward pass
    NetworkWeightsAccum weightsAccum;
    for (unsigned i = 0; i < trace.size(); i++) {
      backprop(trace[i], static_cast<int>(i), memory, weightsAccum);
    }

    // Compile the accumulated weight deltas into a gradient tensor.
    math::Tensor result;
    for (auto &layer : layers) {
      for (auto &weight : layer.weights) {
        Maybe<EMatrix> aw = weightsAccum.GetAccumDelta(weight.first);
        assert(aw.valid()); // During normal training we expect every connection to be updated.

        result.AddLayer(aw.val());
      }
    }

    return result;
  }

  void UpdateWeights(const math::Tensor &weightsDelta) {
    unsigned index = 0;
    for (auto &layer : layers) {
      for (auto &weight : layer.weights) {
        weight.second += weightsDelta(index++);
      }
    }
  }

  void backprop(const SliceBatch &sliceBatch, int timestamp, const LayerMemory &memory,
                NetworkWeightsAccum &weightsAccum) {
    const TimeSlice *networkSlice = memory.GetTimeSlice(timestamp);
    assert(networkSlice != nullptr);

    const LayerMemoryData *outputMemory = networkSlice->GetLayerData(layers.back().layerId);
    assert(outputMemory != nullptr);

    EMatrix outputDelta = outputMemory->output - sliceBatch.batchOutput;
    recursiveBackprop(layers.back(), timestamp, outputDelta, memory, weightsAccum);
  }

  void recursiveBackprop(const Layer &layer, int timestamp, const EMatrix &delta,
                         const LayerMemory &memory, NetworkWeightsAccum &weightsAccum) {

    for (const auto &connection : layer.weights) {
      int nextTimestamp = timestamp + connection.first.timeOffset;
      const TimeSlice *srcSlice = memory.GetTimeSlice(nextTimestamp);
      if (srcSlice == nullptr) {
        continue;
      }

      int batchSize = srcSlice->networkInput.cols();
      if (connection.first.srcLayerId == 0) { // The source is the input.
        EMatrix inputT = getInputWithBias(srcSlice->networkInput).transpose();
        weightsAccum.IncrementWeights(connection.first, delta * inputT * (1.0f / batchSize));
      } else { // The source is another layer from the srcSlice.
        const Layer &srcLayer = findLayer(connection.first.srcLayerId);
        const LayerMemoryData *lmd = srcSlice->GetLayerData(connection.first.srcLayerId);
        assert(lmd != nullptr && lmd->haveOutput);

        EMatrix inputT = getInputWithBias(lmd->output).transpose();
        weightsAccum.IncrementWeights(connection.first, delta * inputT * (1.0f / batchSize));

        // Now compute the delta for the src layer.
        int nRows = connection.second.rows();
        int nCols = connection.second.cols() - 1;
        EMatrix noBiasWeights = connection.second.bottomLeftCorner(nRows, nCols);
        EMatrix srcDelta = noBiasWeights.transpose() * delta;
        assert(srcLayer.numNodes == srcDelta.rows());

        componentScale(srcDelta, lmd->derivative);
        recursiveBackprop(srcLayer, nextTimestamp, srcDelta, memory, weightsAccum);
      }
    }
  }

  void componentScale(EMatrix &target, const EMatrix &scale) {
    assert(target.rows() == scale.rows());
    assert(target.cols() == scale.cols());

    for (int c = 0; c < target.cols(); c++) {
      for (int r = 0; r < target.rows(); r++) {
        target(r, c) = target(r, c) * scale(r, c);
      }
    }
  }

  Layer &findLayer(unsigned layerId) {
    for (auto &layer : layers) {
      if (layer.layerId == layerId) {
        return layer;
      }
    }

    assert(false);
    return layers.back();
  }

  EMatrix forwardPass(const EMatrix &input, const TimeSlice *prevSlice, TimeSlice &curSlice) {
    EMatrix output;
    for (const auto &layer : layers) {
      pair<EMatrix, EMatrix> layerOut = getLayerOutput(layer, prevSlice, curSlice, input);

      LayerMemoryData *lmd = curSlice.GetLayerData(layer.layerId);
      assert(lmd != nullptr);

      lmd->output = layerOut.first;
      lmd->derivative = layerOut.second;
      lmd->haveOutput = true;

      if (layer.isOutput) {
        output = layerOut.first;
      }
    }

    assert(output.rows() == spec.numOutputs);
    assert(output.cols() == input.cols());
    return output;
  }

  // Returns the output vector of the layer, and the derivative vector for the layer.
  pair<EMatrix, EMatrix> getLayerOutput(const Layer &layer, const TimeSlice *prevSlice,
                                        const TimeSlice &curSlice, const EMatrix &networkInput) {
    EMatrix incoming(layer.numNodes, networkInput.cols());
    incoming.fill(0.0f);

    for (const auto &connection : layer.weights) {
      incrementIncomingWithConnection(connection, prevSlice, curSlice, networkInput, incoming);
    }

    return performLayerActivations(layer, incoming);
  }

  void incrementIncomingWithConnection(const pair<LayerConnection, EMatrix> &connection,
                                       const TimeSlice *prevSlice, const TimeSlice &curSlice,
                                       const EMatrix &networkInput, EMatrix &incoming) {

    if (connection.first.srcLayerId == 0) { // special case for input
      assert(connection.first.timeOffset == 0);
      incoming += connection.second * getInputWithBias(networkInput);
    } else {
      const LayerMemoryData *layerMemory = nullptr;

      if (connection.first.timeOffset == 0) {
        layerMemory = curSlice.GetLayerData(connection.first.srcLayerId);
        assert(layerMemory != nullptr);
      } else if (prevSlice != nullptr) {
        layerMemory = prevSlice->GetLayerData(connection.first.srcLayerId);
        assert(layerMemory != nullptr);
      }

      if (layerMemory != nullptr) {
        assert(layerMemory->haveOutput);
        incoming += connection.second * getInputWithBias(layerMemory->output);
      }
    }
  }

  pair<EMatrix, EMatrix> performLayerActivations(const Layer &layer, const EMatrix &incoming) {
    EMatrix activation(incoming.rows(), incoming.cols());
    EMatrix derivatives(incoming.rows(), incoming.cols());

    if (layer.isOutput && spec.outputActivation == LayerActivation::SOFTMAX) {
      for (int c = 0; c < activation.cols(); c++) {
        activation.col(c) = math::SoftmaxActivations(incoming.col(c));
      }
    } else {
      for (int c = 0; c < activation.cols(); c++) {
        for (int r = 0; r < activation.rows(); r++) {
          activation(r, c) = ActivationValue(spec.hiddenActivation, incoming(r, c));
          derivatives(r, c) =
              ActivationDerivative(spec.hiddenActivation, incoming(r, c), activation(r, c));
        }
      }
    }

    return make_pair(activation, derivatives);
  }

  EMatrix getInputWithBias(const EMatrix &noBiasInput) const {
    EMatrix result(noBiasInput.rows() + 1, noBiasInput.cols());
    result.topRightCorner(noBiasInput.rows(), result.cols()) = noBiasInput;
    result.bottomRightCorner(1, result.cols()).fill(1.0f);
    return result;
  }
};

RNN::RNN(const RNNSpec &spec) : impl(new RNNImpl(spec)) {}
RNN::~RNN() = default;

void RNN::ClearMemory(void) { impl->ClearMemory(); }
EMatrix RNN::Process(const EMatrix &input) { return impl->Process(input); }

math::Tensor RNN::ComputeGradient(const vector<SliceBatch> &trace) {
  return impl->ComputeGradient(trace);
}

void RNN::UpdateWeights(const math::Tensor &weightsDelta) {
  return impl->UpdateWeights(weightsDelta);
}
