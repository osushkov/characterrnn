
#include "RNNTrainer.hpp"

using namespace neuralnetwork;
using namespace neuralnetwork::rnn;

static constexpr unsigned TRAINING_SIZE = 10 * 1000 * 1000;
static constexpr unsigned BATCH_SIZE = 100;

struct RNNTrainer::RNNTrainerImpl {
  unsigned traceLength;

  RNNTrainerImpl(unsigned traceLength) : traceLength(traceLength) {}

  uptr<RNN> TrainLanguageNetwork(CharacterStream &cStream, unsigned iters) {
    uptr<RNN> network = createNewNetwork(cStream.VectorDimension(), cStream.VectorDimension());

    vector<math::OneHotVector> letters = cStream.ReadCharacters(TRAINING_SIZE);
    for (unsigned i = 0; i < iters; i++) {
      vector<SliceBatch> batch = makeBatch(letters);
      math::Tensor gradient = updateGradient(network->ComputeGradient(batch));
      network->UpdateWeights(gradient);
    }

    return move(network);
  }

  math::Tensor updateGradient(const math::Tensor &gradient) { return gradient; }

  vector<SliceBatch> makeBatch(const vector<math::OneHotVector> &trainingData) {
    return vector<SliceBatch>();
  }

  uptr<RNN> createNewNetwork(unsigned inputSize, unsigned outputSize) {
    RNNSpec spec;

    spec.numInputs = inputSize;
    spec.numOutputs = outputSize;
    spec.hiddenActivation = LayerActivation::TANH;
    spec.outputActivation = LayerActivation::SOFTMAX;
    spec.nodeActivationRate = 1.0f;

    // Connect layer 1 to the input.
    LayerConnection lc_input_1(0, 1, 0);

    // Connection layer 1 to layer 2, layer 2 to the output layer.
    LayerConnection lc_1_2(1, 2, 0);
    LayerConnection lc_2_output(2, 3, 0);

    // Recurrent self-connections for layers 1 and 2.
    LayerConnection lc_r_11(1, 1, -1);
    LayerConnection lc_r_22(2, 2, -1);

    spec.layers.emplace_back(1, 64, false, vector<LayerConnection>{lc_input_1, lc_r_11});
    spec.layers.emplace_back(2, 32, false, vector<LayerConnection>{lc_1_2, lc_r_22});
    spec.layers.emplace_back(3, outputSize, true, vector<LayerConnection>{lc_2_output});

    return make_unique<RNN>(spec);
  }
};

RNNTrainer::RNNTrainer(unsigned traceLength) : impl(new RNNTrainerImpl(traceLength)) {}

RNNTrainer::~RNNTrainer() = default;

uptr<RNN> RNNTrainer::TrainLanguageNetwork(CharacterStream &cStream, unsigned iters) {
  return impl->TrainLanguageNetwork(cStream, iters);
}
