
#include "LayerMemory.hpp"
#include <cassert>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

LayerMemory::LayerMemory(unsigned historyLength)
    : memory(historyLength), head(0), tail(0), lastTimestamp(-1) {
  assert(historyLength > 0);
  for (auto &m : memory) {
    m.timestamp = -1;
  }
}

LayerMemoryData *LayerMemory::GetLayerData(int timestamp, unsigned layerId) {
  assert(timestamp >= 0);

  for (auto &ts : memory) {
    if (ts.timestamp == timestamp) {
      for (auto &lmd : ts.layers) {
        if (lmd.layerId == layerId) {
          return &lmd;
        }
      }
      return nullptr;
    }
  }

  return nullptr;
}

void LayerMemory::PushNewSlice(const TimeSlice &slice) {
  assert(slice.timestamp >= 0);
  assert(slice.timestamp > lastTimestamp);

  lastTimestamp = slice.timestamp;
  memory[tail] = slice;

  tail = (tail + 1) % memory.size();
  if (tail == head) {
    head = (head + 1) % memory.size();
  }
}

void LayerMemory::Clear(void) {
  head = 0;
  tail = 0;
  lastTimestamp = -1;

  for (auto &m : memory) {
    m.timestamp = -1;
  }
}
