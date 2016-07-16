
#pragma once

#include "../NetworkSpec.hpp"
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct LayerConnection {
  unsigned srcLayerId;
  unsigned dstLayerId;

  int timeOffset; // should be <= 0

  bool operator==(const LayerConnection &other) {
    return this->srcLayerId == other.srcLayerId && this->dstLayerId == other.dstLayerId &&
           this->timeOffset == other.timeOffset;
  }
};

struct LayerSpec {
  unsigned uid;
  unsigned numNodes;
  bool isOutput;

  std::vector<LayerConnection> inConnections;
};

struct RNNSpec {
  unsigned numInputs;
  unsigned numOutputs;
  std::vector<LayerSpec> layers;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;

  float nodeActivationRate; // for dropout regularization.
};
}
}
