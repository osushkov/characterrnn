
#pragma once

#include "../../common/Common.hpp"
#include "../../math/Math.hpp"
#include "TimeSlice.hpp"
#include <vector>

namespace neuralnetwork {
namespace rnn {

class LayerMemory {
public:
  LayerMemory();

  const TimeSlice *GetTimeSlice(int timestamp) const;
  TimeSlice *GetTimeSlice(int timestamp);

  TimeSlice *PushNewSlice(const TimeSlice &slice);

private:
  vector<TimeSlice> memory;
  int lastTimestamp;
};
}
}
