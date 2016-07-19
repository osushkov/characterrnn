
#include "Layer.hpp"
#include "../../math/Math.hpp"
#include <cassert>
#include <cmath>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

static unsigned numLayerOutputs(const RNNSpec &spec, unsigned layerId) {
  if (layerId == 0) {
    return spec.numInputs;
  }

  for (const auto &ls : spec.layers) {
    if (ls.uid == layerId) {
      return ls.numNodes;
    }
  }

  assert(false);
  return 0;
}

static EMatrix createWeightsMatrix(unsigned inputSize, unsigned outputSize) {
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

Layer::Layer(const RNNSpec &nnSpec, const LayerSpec &layerSpec)
    : layerId(layerSpec.uid),
      activation(layerSpec.isOutput ? nnSpec.outputActivation : nnSpec.hiddenActivation),
      numNodes(layerSpec.numNodes), isOutput(layerSpec.isOutput) {

  assert(layerSpec.uid != 0); // 0 is reserved for the input.
  if (isOutput) {
    assert(numNodes == nnSpec.numOutputs);
  }

  for (const auto &lc : layerSpec.inConnections) {
    assert(lc.dstLayerId == layerId);
    assert(lc.timeOffset == 0 || lc.timeOffset == -1);

    // +1 accounts for the bias.
    unsigned inputSize = numLayerOutputs(nnSpec, lc.srcLayerId) + 1;
    EMatrix weightsMatrix = createWeightsMatrix(inputSize, numNodes);
    weights.emplace_back(lc, weightsMatrix);

    // EMatrix accumMatrix = weightsMatrix;
    // accumMatrix.fill(0.0f);
    // layer.accumulatedGradient.push_back(accumMatrix);
    // layer.numSamples.push_back(0);
  }
}
