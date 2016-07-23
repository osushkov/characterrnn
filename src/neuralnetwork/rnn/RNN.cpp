
#include "RNN.hpp"
#include "../../common/Maybe.hpp"
#include "../Activations.hpp"
#include "DeltaAccum.hpp"
#include "GradientAccum.hpp"
#include "Layer.hpp"
#include "LayerMemory.hpp"
#include <cassert>
#include <utility>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

struct BackpropContext {
  LayerMemory memory;
  DeltaAccum deltaAccum;
  GradientAccum gradientAccum;
};

struct RNN::RNNImpl {
  RNNSpec spec;
  vector<Layer> layers;
  Maybe<TimeSlice> previous;

  float softmaxTemperature;

  RNNImpl(const RNNSpec &spec)
      : spec(spec), previous(Maybe<TimeSlice>::none), softmaxTemperature(1.0f) {
    for (const auto &ls : spec.layers) {
      layers.emplace_back(spec, ls);
    }
  }

  RNNImpl(const RNNImpl &other)
      : spec(other.spec), layers(other.layers), previous(other.previous) {}

  void ClearMemory(void) { previous = Maybe<TimeSlice>::none; }

  EMatrix Process(const EMatrix &input, float softmaxTemperature) {
    assert(input.rows() == spec.numInputs);
    assert(input.cols() > 0);

    this->softmaxTemperature = softmaxTemperature;

    TimeSlice *prevSlice = previous.valid() ? &(previous.val()) : nullptr;
    TimeSlice curSlice(0, input, layers);

    EMatrix output = forwardPass(input, prevSlice, curSlice);
    previous = Maybe<TimeSlice>(curSlice);
    return output;
  }

  math::Tensor ComputeGradient(const vector<SliceBatch> &trace) {
    assert(trace.size() > 0);
    assert(trace.front().batchInput.cols() > 0);

    BackpropContext bpContext;

    vector<EMatrix> traceOutputs;
    traceOutputs.reserve(trace.size());

    // Forward pass
    TimeSlice *prevSlice = nullptr;
    for (unsigned i = 0; i < trace.size(); i++) {
      TimeSlice curSlice(i, trace[i].batchInput, layers);

      EMatrix out = forwardPass(trace[i].batchInput, prevSlice, curSlice);
      traceOutputs.push_back(out);

      prevSlice = bpContext.memory.PushNewSlice(curSlice);
    }

    assert(trace.size() == traceOutputs.size());

    // Backward pass
    float totalLoss = 0.0f;
    for (int i = (trace.size() - 1); i >= 0; i--) {
      totalLoss += backprop(trace[i], i, bpContext);
    }

    // Compile the accumulated weight deltas into a gradient tensor.
    float batchScale = 1.0f / static_cast<float>(trace.front().batchInput.cols());

    math::Tensor result;
    for (auto &layer : layers) {
      for (auto &weight : layer.weights) {
        Maybe<EMatrix> aw = bpContext.gradientAccum.GetGradient(weight.first);
        assert(aw.valid()); // During normal training we expect every connection to be updated.

        result.AddLayer(aw.val() * batchScale);
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

  float backprop(const SliceBatch &sliceBatch, int timestamp, BackpropContext &bpContext) {
    const TimeSlice *networkSlice = bpContext.memory.GetTimeSlice(timestamp);
    assert(networkSlice != nullptr);

    const LayerMemoryData *outputMemory = networkSlice->GetLayerData(layers.back().layerId);
    assert(outputMemory != nullptr);
    assert(outputMemory->haveOutput);

    EMatrix outputDelta = outputMemory->output - sliceBatch.batchOutput;
    bpContext.deltaAccum.IncrementDelta(layers.back().layerId, timestamp, outputDelta);

    recursiveBackprop(layers.back(), timestamp, outputDelta, bpContext);

    float loss = 0.0f;
    for (int r = 0; r < outputDelta.rows(); r++) {
      for (int c = 0; c < outputDelta.cols(); c++) {
        loss += outputDelta(r, c) * outputDelta(r, c);
      }
    }
    return loss;
  }

  void recursiveBackprop(const Layer &layer, int timestamp, const EMatrix &delta,
                         BackpropContext &bpContext) {

    for (const auto &connection : layer.weights) {
      int srcTimestamp = timestamp - connection.first.timeOffset;
      // cout << "recursive backprop: " << layer.layerId << " (" << timestamp << ") -> " <<
      // connection.first.srcLayerId << " (" << srcTimestamp << ")" << endl;

      const TimeSlice *srcSlice = bpContext.memory.GetTimeSlice(srcTimestamp);
      if (srcSlice == nullptr) {
        continue;
      }

      if (connection.first.srcLayerId == 0) { // The source is the input.
        EMatrix inputT = getInputWithBias(srcSlice->networkInput).transpose();
        bpContext.gradientAccum.IncrementWeights(connection.first, delta * inputT);
        // cout << "increment weight: " << connection.first.srcLayerId << "-"
        // << connection.first.dstLayerId << endl;
      } else { // The source is another layer from the srcSlice.
        const Layer &srcLayer = findLayer(connection.first.srcLayerId);
        const LayerMemoryData *lmd = srcSlice->GetLayerData(srcLayer.layerId);
        assert(lmd != nullptr && lmd->haveOutput);

        // Accumulate the gradient for the connection weight.
        EMatrix inputT = getInputWithBias(lmd->output).transpose();
        bpContext.gradientAccum.IncrementWeights(connection.first, delta * inputT);
        // cout << "increment weight: " << connection.first.srcLayerId << "-"
        // << connection.first.dstLayerId << endl;

        // Now increment the delta for the src layer.
        int nRows = connection.second.rows();
        int nCols = connection.second.cols() - 1;
        EMatrix noBiasWeights = connection.second.bottomLeftCorner(nRows, nCols);
        EMatrix srcDelta = noBiasWeights.transpose() * delta;
        assert(srcLayer.numNodes == srcDelta.rows());

        componentScale(srcDelta, lmd->derivative);
        LayerAccum &deltaAccum =
            bpContext.deltaAccum.IncrementDelta(srcLayer.layerId, srcTimestamp, srcDelta);

        if (connection.first.timeOffset == 0) {
          recursiveBackprop(srcLayer, srcTimestamp, deltaAccum.GetDelta(), bpContext);
        }
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
        activation.col(c) = math::SoftmaxActivations(incoming.col(c) / softmaxTemperature);
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
RNN::RNN(const RNN &other) : impl(new RNNImpl(*other.impl)) {}

RNN::~RNN() = default;

RNN &RNN::operator=(const RNN &other) {
  impl.reset(new RNNImpl(*other.impl));
  return *this;
}

void RNN::ClearMemory(void) { impl->ClearMemory(); }
EMatrix RNN::Process(const EMatrix &input, float softmaxTemperature) {
  return impl->Process(input, softmaxTemperature);
}

math::Tensor RNN::ComputeGradient(const vector<SliceBatch> &trace) {
  return impl->ComputeGradient(trace);
}

void RNN::UpdateWeights(const math::Tensor &weightsDelta) {
  return impl->UpdateWeights(weightsDelta);
}
