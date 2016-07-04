#pragma once

#include "CharacterStream.hpp"
#include "common/Common.hpp"
#include "neuralnetwork/Network.hpp"

class FFNetworkTrainer {
public:
  FFNetworkTrainer(unsigned nGramSize);
  ~FFNetworkTrainer();

  uptr<neuralnetwork::Network> TrainLanguageNetwork(CharacterStream &cStream, unsigned iters);

private:
  struct FFNetworkTrainerImpl;
  uptr<FFNetworkTrainerImpl> impl;
};
