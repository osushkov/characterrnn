
#pragma once

#include "../../common/Common.hpp"
#include "../../math/Math.hpp"
#include "../../math/Tensor.hpp"
#include "../SamplesProvider.hpp"
#include "RNNSpec.hpp"

namespace neuralnetwork {
namespace rnn {

struct SliceBatch {
  EMatrix batchInput;
  EMatrix batchOutput;

  SliceBatch(const EMatrix &batchInput, const EMatrix &batchOutput)
      : batchInput(batchInput), batchOutput(batchOutput) {}
};

class RNN {
public:
  RNN(const RNNSpec &spec);
  RNN(const RNN &other);
  virtual ~RNN();

  RNN &operator=(const RNN &other);

  void ClearMemory(void);
  EMatrix Process(const EMatrix &input, float softmaxTemperature);

  math::Tensor ComputeGradient(const vector<SliceBatch> &trace);
  void UpdateWeights(const math::Tensor &weightsDelta);

private:
  struct RNNImpl;
  uptr<RNNImpl> impl;
};
}
}
