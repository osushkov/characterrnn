
#include "RNN.hpp"
#include "LayerMemory.hpp"
#include <cassert>
#include <utility>

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

static constexpr unsigned LEARNING_HISTORY_LENGTH = 10;

struct Layer {
  unsigned layerId;
  LayerActivation activation;

  unsigned numNodes;

  // Weights for incoming connections from other layers.
  vector<pair<LayerConnection, EMatrix>> weights;
};

struct RNN::RNNImpl {
  RNNSpec spec;
  LayerMemory memory;
  vector<Layer> layers;

  RNNImpl(const RNNSpec &spec) : spec(spec), memory(LEARNING_HISTORY_LENGTH) { initialiseLayers(); }

  void ClearMemory(void) { memory.Clear(); }

  EVector Process(const EVector &input) { return input; }

  void Update(const SamplesProvider &samplesProvider) {}

  void initialiseLayers(void) {}
};

RNN::RNN(const RNNSpec &spec) : impl(new RNNImpl(spec)) {}
RNN::~RNN() = default;

void RNN::ClearMemory(void) { impl->ClearMemory(); }
EVector RNN::Process(const EVector &input) { return impl->Process(input); }
void RNN::Update(const SamplesProvider &samplesProvider) { impl->Update(samplesProvider); }
