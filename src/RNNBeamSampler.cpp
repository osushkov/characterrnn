#include "RNNBeamSampler.hpp"
#include <algorithm>
#include <cassert>

static constexpr unsigned NUM_BEAMS = 100;
static constexpr unsigned RESAMPLE_RATE = 10;
static constexpr unsigned RESAMPLE_DROP = 30;

struct SampledBeam {
  neuralnetwork::rnn::RNN rnn;
  unsigned letterDim;

  float logProbabilitySum;
  vector<unsigned> samples;

  SampledBeam(const neuralnetwork::rnn::RNN &rnn, unsigned letterDim)
      : rnn(rnn), letterDim(letterDim), logProbabilitySum(0.0) {}

  void Sample(void) {
    EVector input(letterDim);
    input.fill(0.0f);

    if (samples.size() > 0) {
      input(samples.back()) = 1.0f;
    }

    EVector pChar = rnn.Process(input, 1.0);
    float r = math::UnitRand();

    for (int i = 0; i < pChar.rows(); i++) {
      r -= pChar(i);
      if (r < 0.0f) {
        logProbabilitySum += logf(pChar(i));
        samples.push_back(i);
        return;
      }
    }

    unsigned index = rand() % pChar.rows();
    logProbabilitySum += logf(pChar(index));
    samples.push_back(index);
  }
};

struct RNNBeamSampler::RNNBeamSamplerImpl {
  unsigned letterDim;

  RNNBeamSamplerImpl(unsigned letterDim) : letterDim(letterDim) { assert(letterDim > 0); }

  vector<unsigned> SampleCharacters(neuralnetwork::rnn::RNN *network, unsigned numChars) {
    network->ClearMemory();

    vector<SampledBeam> beams;
    for (unsigned i = 0; i < NUM_BEAMS; i++) {
      beams.emplace_back(*network, letterDim);
    }

    for (unsigned i = 0; i < numChars; i++) {
      for (auto &beam : beams) {
        beam.Sample();
      }

      if (i % RESAMPLE_RATE == 0) {
        sort(beams.begin(), beams.end(), [](const SampledBeam &a, const SampledBeam &b) {
          return a.logProbabilitySum < b.logProbabilitySum;
        });

        for (unsigned j = 0; j < RESAMPLE_DROP; j++) {
          beams[j] = beams[RESAMPLE_DROP + (rand() % (NUM_BEAMS - RESAMPLE_DROP))];
        }
      }
    }

    sort(beams.begin(), beams.end(), [](const SampledBeam &a, const SampledBeam &b) {
      return a.logProbabilitySum < b.logProbabilitySum;
    });

    cout << beams.front().logProbabilitySum << " : " << beams.back().logProbabilitySum << endl;
    return beams.back().samples;
  }
};

RNNBeamSampler::RNNBeamSampler(unsigned letterDim) : impl(new RNNBeamSamplerImpl(letterDim)) {}

RNNBeamSampler::~RNNBeamSampler() = default;

vector<unsigned> RNNBeamSampler::SampleCharacters(neuralnetwork::rnn::RNN *network,
                                                  unsigned numChars) {
  return impl->SampleCharacters(network, numChars);
}
