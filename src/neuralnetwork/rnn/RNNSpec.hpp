
#pragma once

#include "../../common/Common.hpp"
#include "../NetworkSpec.hpp"
#include <cassert>
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct LayerConnection {
  unsigned srcLayerId;
  unsigned dstLayerId;

  int timeOffset; // should be 0 or 1

  LayerConnection(unsigned srcLayerId, unsigned dstLayerId, int timeOffset)
      : srcLayerId(srcLayerId), dstLayerId(dstLayerId), timeOffset(timeOffset) {
    assert(timeOffset == 0 || timeOffset == 1);
  }

  bool operator==(const LayerConnection &other) const {
    return this->srcLayerId == other.srcLayerId && this->dstLayerId == other.dstLayerId &&
           this->timeOffset == other.timeOffset;
  }
};

struct LayerSpec {
  unsigned uid; // must be >= 1, 0 is the "input" layer.
  unsigned numNodes;
  bool isOutput;

  LayerSpec(unsigned uid, unsigned numNodes, bool isOutput)
      : uid(uid), numNodes(numNodes), isOutput(isOutput) {
    assert(uid >= 1);
    assert(numNodes > 0);
  }
};

struct RNNSpec {
  unsigned numInputs;
  unsigned numOutputs;
  vector<LayerSpec> layers;
  vector<LayerConnection> connections;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;

  float nodeActivationRate; // for dropout regularization.
};
}
}
