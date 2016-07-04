
#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "NetworkSpec.hpp"
#include "SamplesProvider.hpp"
#include <vector>

namespace neuralnetwork {

class Network {
public:
  Network(const NetworkSpec &spec);
  virtual ~Network();

  EVector Process(const EVector &input); // TODO: make this const?
  void Refresh(void);
  void Update(const SamplesProvider &samplesProvider);

private:
  // Non-copyable
  Network(const Network &other) = delete;
  Network &operator=(const Network &) = delete;

  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
}
