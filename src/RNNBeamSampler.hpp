#pragma once

#include "common/Common.hpp"
#include "neuralnetwork/rnn/RNN.hpp"
#include <vector>

class RNNBeamSampler {
public:
  RNNBeamSampler(unsigned letterDim);
  ~RNNBeamSampler();

  vector<unsigned> SampleCharacters(neuralnetwork::rnn::RNN *network, unsigned numChars);

private:
  struct RNNBeamSamplerImpl;
  uptr<RNNBeamSamplerImpl> impl;
};
