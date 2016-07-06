
#pragma once

#include "../../common/Common.hpp"
#include "../../math/Math.hpp"
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct LayerMemoryData {
  EMatrix output; // batch output, column per batch element.
  EMatrix derivative;
  EMatrix delta;

  unsigned layerId;
};

struct TimeSlice {
  int timestamp;
  vector<LayerMemoryData> layers;
};

class LayerMemory {
public:
  LayerMemory(unsigned historyLength);

  LayerMemoryData *GetLayerData(int timestamp, unsigned layerId);
  void PushNewSlice(const TimeSlice &slice);
  void Clear(void);

private:
  vector<TimeSlice> memory;
  unsigned head;
  unsigned tail;

  int lastTimestamp;
};
}
}
