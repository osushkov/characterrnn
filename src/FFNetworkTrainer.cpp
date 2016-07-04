
#include "FFNetworkTrainer.hpp"
#include "neuralnetwork/SamplesProvider.hpp"
#include "neuralnetwork/TrainingSample.hpp"
#include <cassert>

using namespace neuralnetwork;

static constexpr unsigned NUM_LETTERS = 10 * 1000 * 1000;
static constexpr unsigned BATCH_SIZE = 500;

struct FFNetworkTrainer::FFNetworkTrainerImpl {
  unsigned nGramSize;

  FFNetworkTrainerImpl(unsigned nGramSize) : nGramSize(nGramSize) { assert(nGramSize >= 1); }

  uptr<Network> TrainLanguageNetwork(CharacterStream &cStream, unsigned iters) {
    uptr<Network> network =
        createNewNetwork(nGramSize * cStream.VectorDimension(), cStream.VectorDimension());

    vector<math::OneHotVector> letters = cStream.ReadCharacters(NUM_LETTERS);
    vector<TrainingSample> allSamples = makeTrainingSamples(letters, cStream.VectorDimension());
    random_shuffle(allSamples.begin(), allSamples.end());

    for (unsigned i = 0; i < iters; i++) {
      auto samplesProvider = SamplesProvider(allSamples, BATCH_SIZE, rand() % allSamples.size());
      network->Update(samplesProvider);

      if (i % 100 == 0) {
        unsigned percentDone = (100 * i) / iters;
        cout << percentDone << "%" << endl;
      }
    }

    network->Refresh();
    return move(network);
  }

  vector<TrainingSample> makeTrainingSamples(const vector<math::OneHotVector> &lettersStream,
                                             unsigned dim) {
    EVector zero(dim);
    zero.fill(0.0f);

    vector<EVector> prevLetters(nGramSize, zero);
    unsigned head = 0;
    unsigned tail = prevLetters.size() - 1;

    vector<TrainingSample> result;
    result.reserve(NUM_LETTERS);

    for (const auto &letter : lettersStream) {
      EVector letterVec = zero;
      letterVec(letter.index) = 1.0f;

      result.push_back(sampleFromLetters(prevLetters, head, letterVec));

      head = (head + 1) % prevLetters.size();
      tail = (tail + 1) % prevLetters.size();
      prevLetters[tail] = letterVec;
    }

    return result;
  }

  TrainingSample sampleFromLetters(const vector<EVector> &prevLetters, unsigned head,
                                   const EVector &curLetter) {
    assert(prevLetters.size() > 0);

    EVector input(prevLetters.size() * prevLetters[0].rows());
    unsigned j = 0;

    for (unsigned i = 0; i < prevLetters.size(); i++) {
      unsigned index = (head + i) % prevLetters.size();
      for (int lr = 0; lr < prevLetters[index].rows(); lr++) {
        input(j++) = prevLetters[index](lr);
      }
    }

    return TrainingSample(input, curLetter);
  }

  uptr<Network> createNewNetwork(unsigned inputSize, unsigned outputSize) {
    NetworkSpec spec;
    spec.numInputs = inputSize;
    spec.numOutputs = outputSize;
    spec.hiddenLayers = {inputSize, inputSize / 2, inputSize / 4};
    spec.nodeActivationRate = 1.0f;
    spec.maxBatchSize = BATCH_SIZE;
    spec.hiddenActivation = LayerActivation::TANH;
    spec.outputActivation = LayerActivation::SOFTMAX;

    return make_unique<Network>(spec);
  }
};

FFNetworkTrainer::FFNetworkTrainer(unsigned nGramSize)
    : impl(new FFNetworkTrainerImpl(nGramSize)) {}

FFNetworkTrainer::~FFNetworkTrainer() = default;

uptr<Network> FFNetworkTrainer::TrainLanguageNetwork(CharacterStream &cStream, unsigned iters) {
  return impl->TrainLanguageNetwork(cStream, iters);
}
