#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/Network.hpp"
#include <vector>

class FFNetworkSampler {
public:
  FFNetworkSampler(unsigned nGramSize, unsigned letterDim);
  ~FFNetworkSampler();

  vector<unsigned> SampleCharacters(neuralnetwork::Network *network, unsigned numChars);

private:
  struct FFNetworkSamplerImpl;
  uptr<FFNetworkSamplerImpl> impl;
};
