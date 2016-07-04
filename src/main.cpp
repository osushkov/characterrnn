
#include "CharacterStream.hpp"
#include "FFNetworkSampler.hpp"
#include "FFNetworkTrainer.hpp"
#include "common/Common.hpp"

static constexpr unsigned NGRAM_SIZE = 4;

int main(int argc, char **argv) {
  srand(1234);

  string path(argv[1]);
  CharacterStream cstream(path);

  FFNetworkTrainer trainer(NGRAM_SIZE);
  auto network = trainer.TrainLanguageNetwork(cstream, 100000);

  FFNetworkSampler sampler(NGRAM_SIZE, cstream.VectorDimension());
  vector<unsigned> sampled = sampler.SampleCharacters(network.get(), 1000);

  for (const auto sample : sampled) {
    cout << cstream.Decode(sample);
  }

  cout << endl;
  return 0;
}
