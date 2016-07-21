#pragma once

#include "../../common/Common.hpp"
#include "../../math/Math.hpp"
#include "../Activations.hpp"
#include "RNNSpec.hpp"
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct Layer {
  unsigned layerId;
  LayerActivation activation;

  unsigned numNodes;
  bool isOutput;

  // Weights for incoming connections from other layers.
  vector<pair<LayerConnection, EMatrix>> weights;
  unsigned numOutgoingConnections;

  Layer(const RNNSpec &nnSpec, const LayerSpec &layerSpec);
};
}
}
