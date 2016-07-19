
#include "LayerMemory.hpp"
#include <cassert>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

LayerMemory::LayerMemory() : lastTimestamp(-1) {}

// TODO: use const_cast to avoid duplication.
const TimeSlice *LayerMemory::GetTimeSlice(int timestamp) const {
  for (auto &ts : memory) {
    if (ts.timestamp == timestamp) {
      return &ts;
    }
  }

  return nullptr;
}

TimeSlice *LayerMemory::GetTimeSlice(int timestamp) {
  for (auto &ts : memory) {
    if (ts.timestamp == timestamp) {
      return &ts;
    }
  }

  return nullptr;
}

TimeSlice *LayerMemory::PushNewSlice(const TimeSlice &slice) {
  assert(slice.timestamp >= 0);
  assert(slice.timestamp > lastTimestamp);

  memory.push_back(slice);
  lastTimestamp = slice.timestamp;

  return &memory.back();
}
