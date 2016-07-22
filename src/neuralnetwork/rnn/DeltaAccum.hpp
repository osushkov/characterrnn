#pragma once

#include "../../common/Common.hpp"
#include "../../common/Maybe.hpp"
#include "../../math/Math.hpp"
#include "Layer.hpp"
#include "RNNSpec.hpp"
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct LayerAccum {
  unsigned layerId;
  int timestamp;

  unsigned samples;
  EMatrix accumDelta;

  LayerAccum(unsigned layerId, int timestamp)
      : layerId(layerId), timestamp(timestamp), samples(0) {}

  EMatrix GetDelta(void) const {
    assert(samples > 0);
    return accumDelta * (1.0f / static_cast<float>(samples));
  }

  void AccumDelta(const EMatrix &delta) {
    if (samples == 0) {
      accumDelta = delta;
    } else {
      accumDelta += delta;
    }

    samples++;
  }
};

struct DeltaAccum {
  vector<LayerAccum> allDeltaAccum;

  LayerAccum &IncrementDelta(unsigned layerId, int timestamp, const EMatrix &delta) {
    for (auto &da : allDeltaAccum) {
      if (da.layerId == layerId && da.timestamp == timestamp) {
        da.AccumDelta(delta);
        return da;
      }
    }

    allDeltaAccum.emplace_back(layerId, timestamp);
    allDeltaAccum.back().AccumDelta(delta);
    return allDeltaAccum.back();
  }

  LayerAccum &GetDelta(unsigned layerId, int timestamp) {
    for (auto &da : allDeltaAccum) {
      if (da.layerId == layerId && da.timestamp == timestamp) {
        return da;
      }
    }

    assert(false);
    return allDeltaAccum.front();
  }

  void DebugPrint(void) {
    cout << "num deltas accumulated: " << allDeltaAccum.size() << endl;
    for (const auto &wa : allDeltaAccum) {
      cout << "acc: " << wa.layerId << " , " << wa.timestamp << " = " << wa.samples << endl;
    }
  }
};
}
}
