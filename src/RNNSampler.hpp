#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/rnn/RNN.hpp"
#include <vector>

class RNNSampler {
public:
  RNNSampler(unsigned letterDim);
  ~RNNSampler();

  vector<unsigned> SampleCharacters(neuralnetwork::rnn::RNN *network, unsigned numChars);

private:
  struct RNNSamplerImpl;
  uptr<RNNSamplerImpl> impl;
};
