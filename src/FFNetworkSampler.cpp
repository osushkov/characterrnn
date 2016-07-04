
#include "FFNetworkSampler.hpp"
#include <cassert>

struct FFNetworkSampler::FFNetworkSamplerImpl {
  unsigned nGramSize;
  unsigned letterDim;

  FFNetworkSamplerImpl(unsigned nGramSize, unsigned letterDim)
      : nGramSize(nGramSize), letterDim(letterDim) {
    assert(nGramSize > 0);
    assert(letterDim > 0);
  }

  vector<unsigned> SampleCharacters(neuralnetwork::Network *network, unsigned numChars) {
    vector<unsigned> result;
    result.reserve(numChars);

    for (unsigned i = 0; i < numChars; i++) {
      unsigned sample = sampleChar(network, result);
      result.push_back(sample);
    }

    return result;
  }

  unsigned sampleChar(neuralnetwork::Network *network, const vector<unsigned> &prevChars) {
    EVector input(nGramSize * letterDim);
    input.fill(0.0f);

    for (unsigned i = 0; i < nGramSize; i++) {
      int prevCharsIndex = prevChars.size() - i - 1;
      if (prevCharsIndex < 0) {
        break;
      }

      input((nGramSize - i - 1) * letterDim + prevChars[prevCharsIndex]) = 1.0f;
    }

    EVector pChar = network->Process(input);
    float r = math::UnitRand();

    for (int i = 0; i < pChar.rows(); i++) {
      r -= pChar(i);
      if (r < 0.0f) {
        return static_cast<unsigned>(i);
      }
    }

    return static_cast<unsigned>(rand() % pChar.rows());
  }
};

FFNetworkSampler::FFNetworkSampler(unsigned nGramSize, unsigned letterDim)
    : impl(new FFNetworkSamplerImpl(nGramSize, letterDim)) {}

FFNetworkSampler::~FFNetworkSampler() = default;

vector<unsigned> FFNetworkSampler::SampleCharacters(neuralnetwork::Network *network,
                                                    unsigned numChars) {
  return impl->SampleCharacters(network, numChars);
}
