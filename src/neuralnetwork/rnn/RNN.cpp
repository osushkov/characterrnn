
#include "RNN.hpp"
#include "../Activations.hpp"
#include "LayerMemory.hpp"
#include <cassert>
#include <utility>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

static constexpr unsigned LEARNING_HISTORY_LENGTH = 10;

struct Layer {
  unsigned layerId;
  LayerActivation activation;

  unsigned numNodes;
  bool isOutput;

  // Weights for incoming connections from other layers.
  vector<pair<LayerConnection, EMatrix>> weights;
};

struct RNN::RNNImpl {
  RNNSpec spec;
  LayerMemory memory;
  vector<Layer> layers;
  int curTimestamp;

  RNNImpl(const RNNSpec &spec) : spec(spec), memory(LEARNING_HISTORY_LENGTH), curTimestamp(0) {
    initialiseLayers();
  }

  void ClearMemory(void) {
    memory.Clear();
    curTimestamp = 0;
  }

  EMatrix Process(const EMatrix &input) {
    assert(input.rows() == spec.numInputs);

    memory.PushNewSlice(createNewTimeSlice(input.cols()));

    EMatrix output;
    for (const auto &layer : layers) {
      pair<EMatrix, EMatrix> layerOut = getLayerOutput(layer, input);

      LayerMemoryData *lmd = memory.GetLayerData(curTimestamp, layer.layerId);
      assert(lmd != nullptr);

      lmd->output = layerOut.first;
      lmd->derivative = layerOut.second;

      if (layer.isOutput) {
        output = layerOut.first;
      }
    }

    assert(output.rows() == spec.numOutputs);
    assert(output.cols() == input.cols());

    curTimestamp++;
    return output;
  }

  void Update(const SamplesProvider &samplesProvider) {}

  TimeSlice createNewTimeSlice(int batchSize) {
    TimeSlice result;
    result.timestamp = curTimestamp;

    for (const auto &layer : layers) {
      LayerMemoryData layerData;
      layerData.layerId = layer.layerId;

      layerData.output = EMatrix(layer.numNodes, batchSize);
      layerData.derivative = EMatrix(layer.numNodes, batchSize);
      layerData.delta = EMatrix(layer.numNodes, batchSize);

      layerData.output.fill(0.0f);
      layerData.derivative.fill(0.0f);
      layerData.delta.fill(0.0f);

      result.layers.push_back(layerData);
    }

    return result;
  }

  // Returns the output vector of the layer, and the derivative vector for the layer.
  pair<EMatrix, EMatrix> getLayerOutput(const Layer &layer, const EMatrix &batchInput) {
    EMatrix incoming(layer.numNodes, batchInput.cols());
    incoming.fill(0.0f);

    for (const auto &connection : layer.weights) {
      incrementIncomingWithConnection(connection, batchInput, incoming);
    }

    return performLayerActivations(layer, incoming);
  }

  void incrementIncomingWithConnection(const pair<LayerConnection, EMatrix> &connection,
                                       const EMatrix &batchInput, EMatrix &incoming) {

    if (connection.first.srcLayerId == 0) { // special case for input
      assert(connection.first.timeOffset == 0);
      incoming += connection.second * getInputWithBias(batchInput);
    } else {
      LayerMemoryData *layerMemory = memory.GetLayerData(curTimestamp + connection.first.timeOffset,
                                                         connection.first.srcLayerId);

      assert(layerMemory != nullptr || connection.first.timeOffset == -1);
      if (layerMemory != nullptr) {
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

  void initialiseLayers(void) {
    for (const auto &ls : spec.layers) {
      layers.push_back(createLayer(ls));
    }
  }

  Layer createLayer(const LayerSpec &layerSpec) {
    assert(layerSpec.uid != 0); // 0 is reserved for the input.

    Layer layer;
    layer.layerId = layerSpec.uid;
    layer.activation = layerSpec.isOutput ? spec.outputActivation : spec.hiddenActivation;
    layer.numNodes = layerSpec.numNodes;
    layer.isOutput = layerSpec.isOutput;

    if (layerSpec.isOutput) {
      assert(layer.numNodes == spec.numOutputs);
    }

    for (const auto &lc : layerSpec.inConnections) {
      assert(lc.dstLayerId == layer.layerId);
      assert(lc.timeOffset == 0 || lc.timeOffset == -1);

      // +1 accounts for the bias.
      unsigned inputSize = numLayerOutputs(lc.srcLayerId) + 1;
      layer.weights.emplace_back(lc, createWeightsMatrix(inputSize, layer.numNodes));
    }

    return layer;
  }

  EMatrix createWeightsMatrix(unsigned inputSize, unsigned outputSize) {
    float initRange = 1.0f / sqrtf(inputSize);

    EMatrix result(outputSize, inputSize);
    result.fill(0.0f);

    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < result.cols(); c++) {
        // TODO: consider replacing this with a Gaussian with std-dev of initRange.
        result(r, c) = math::RandInterval(-initRange, initRange);
      }
    }

    return result;
  }

  unsigned numLayerOutputs(unsigned layerId) {
    for (const auto &ls : spec.layers) {
      if (ls.uid == layerId) {
        return ls.numNodes;
      }
    }
    assert(false);
    return 0;
  }
};

RNN::RNN(const RNNSpec &spec) : impl(new RNNImpl(spec)) {}
RNN::~RNN() = default;

void RNN::ClearMemory(void) { impl->ClearMemory(); }
EMatrix RNN::Process(const EMatrix &input) { return impl->Process(input); }
void RNN::Update(const SamplesProvider &samplesProvider) { impl->Update(samplesProvider); }
