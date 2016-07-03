
#include "FFNetworkTrainer.hpp"

struct FFNetworkTrainer::FFNetworkTrainerImpl {
  FFNetworkTrainerImpl(unsigned nGramSize) {}

  uptr<neuralnetwork::Network> TrainLanguageNetwork(CharacterStream &cStream) { return nullptr; }
};

FFNetworkTrainer::FFNetworkTrainer(unsigned nGramSize)
    : impl(new FFNetworkTrainerImpl(nGramSize)) {}

FFNetworkTrainer::~FFNetworkTrainer() = default;

uptr<neuralnetwork::Network> FFNetworkTrainer::TrainLanguageNetwork(CharacterStream &cStream) {
  return impl->TrainLanguageNetwork(cStream);
}
