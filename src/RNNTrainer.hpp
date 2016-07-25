#pragma once

#include "CharacterStream.hpp"
#include "common/Common.hpp"
#include "neuralnetwork/rnn/RNN.hpp"

class RNNTrainer {
public:
  RNNTrainer(unsigned miniTraceLength);
  ~RNNTrainer();

  uptr<neuralnetwork::rnn::RNN> TrainLanguageNetwork(CharacterStream &cStream, unsigned iters);

private:
  struct RNNTrainerImpl;
  uptr<RNNTrainerImpl> impl;
};
