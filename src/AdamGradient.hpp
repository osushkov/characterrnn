#pragma once

#include "math/Tensor.hpp"

class AdamGradient {
public:
  AdamGradient(); // TODO: put in a non-default constructor that allows the params to be set.
  ~AdamGradient() = default;

  math::Tensor UpdateGradient(const math::Tensor &gradient);

private:
  const float beta1;
  const float beta2;
  const float epsilon;
  const float lr;

  math::Tensor momentum;
  math::Tensor rms;
  bool isFirstUpdate;

  math::Tensor initialMomentum(const math::Tensor &gradient);
  math::Tensor initialRMS(const math::Tensor &gradient);

  void updateMomentum(const math::Tensor &gradient);
  void updateRMS(const math::Tensor &gradient);
  math::Tensor computeWeightUpdate(void);
};
