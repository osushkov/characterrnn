
#include "CharacterStream.hpp"
#include "FFNetworkSampler.hpp"
#include "FFNetworkTrainer.hpp"
#include "RNNBeamSampler.hpp"
#include "RNNSampler.hpp"
#include "RNNTrainer.hpp"
#include "common/Common.hpp"
#include "common/Maybe.hpp"
#include "neuralnetwork/rnn/RNN.hpp"

static constexpr unsigned NGRAM_SIZE = 4;

void testFFNetwork(string path) {
  CharacterStream cstream(path);

  FFNetworkTrainer trainer(NGRAM_SIZE);
  auto network = trainer.TrainLanguageNetwork(cstream, 100000);

  FFNetworkSampler sampler(NGRAM_SIZE, cstream.VectorDimension());
  vector<unsigned> sampled = sampler.SampleCharacters(network.get(), 1000);

  for (const auto sample : sampled) {
    cout << cstream.Decode(sample);
  }

  cout << endl;
}

void testRNN(string path) {
  CharacterStream cstream(path);

  RNNTrainer trainer(24);
  auto network = trainer.TrainLanguageNetwork(cstream, 5000000);

  RNNSampler sampler(cstream.VectorDimension());
  // RNNBeamSampler sampler(cstream.VectorDimension());
  vector<unsigned> sampled = sampler.SampleCharacters(network.get(), 10000);

  for (const auto sample : sampled) {
    cout << cstream.Decode(sample);
  }

  cout << endl;
}

int main(int argc, char **argv) {
  srand(1234);

  string path(argv[1]);

  // testFFNetwork(path);
  testRNN(path);

  return 0;
}
