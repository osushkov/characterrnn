#pragma once

#include "../../common/Common.hpp"
#include "../../math/Math.hpp"
#include "Layer.hpp"
#include <cassert>
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct LayerMemoryData {
  EMatrix output; // batch output, column per batch element.
  EMatrix derivative;

  unsigned layerId;
  bool haveOutput;
};

struct TimeSlice {
  int timestamp;
  EMatrix networkInput;
  vector<LayerMemoryData> layerData;

  TimeSlice(int timestamp, const EMatrix &networkInput, const vector<Layer> &layers)
      : timestamp(timestamp), networkInput(networkInput) {
    assert(networkInput.cols() > 0);

    layerData.reserve(layers.size());
    for (const auto &layer : layers) {
      layerData.push_back(LayerMemoryData());

      LayerMemoryData &lmd = layerData.back();
      lmd.layerId = layer.layerId;

      lmd.output = EMatrix(layer.numNodes, networkInput.cols());
      lmd.output.fill(0.0f);

      lmd.derivative = EMatrix(layer.numNodes, networkInput.cols());
      lmd.derivative.fill(0.0f);

      lmd.haveOutput = false;
    }
  }

  const LayerMemoryData *GetLayerData(unsigned layerId) const {
    for (auto &lmd : layerData) {
      if (lmd.layerId == layerId) {
        return &lmd;
      }
    }
    return nullptr;
  }

  // TODO: use const_cast to not have duplicate code.
  LayerMemoryData *GetLayerData(unsigned layerId) {
    for (auto &lmd : layerData) {
      if (lmd.layerId == layerId) {
        return &lmd;
      }
    }
    return nullptr;
  }
};
}
}
