
#include "AdamGradient.hpp"
#include <cassert>

AdamGradient::AdamGradient()
    : beta1(0.9f), beta2(0.999f), epsilon(10e-8), lr(0.001f), isFirstUpdate(false) {}

math::Tensor AdamGradient::UpdateGradient(const math::Tensor &gradient) { return gradient; }

math::Tensor AdamGradient::initialMomentum(const math::Tensor &gradient) {
  math::Tensor result = gradient;
  for (unsigned i = 0; i < result.NumLayers(); i++) {
    result(i).fill(0.0f);
  }
  return result;
}

math::Tensor AdamGradient::initialRMS(const math::Tensor &gradient) {
  math::Tensor result = gradient;
  for (unsigned i = 0; i < result.NumLayers(); i++) {
    result(i).fill(0.0f);
  }
  return result;
}

void AdamGradient::updateMomentum(math::Tensor &momentum, const math::Tensor &gradient) {
  assert(gradient.NumLayers() == momentum.NumLayers());
  momentum = momentum * beta1 + gradient * (1.0f - beta1);
}

void AdamGradient::updateRMS(math::Tensor &rms, const math::Tensor &gradient) {
  assert(gradient.NumLayers() == rms.NumLayers());

  for (unsigned i = 0; i < gradient.NumLayers(); i++) {
    assert(gradient(i).rows() == rms(i).rows());
    assert(gradient(i).cols() == rms(i).cols());

    for (int y = 0; y < gradient(i).rows(); y++) {
      for (int x = 0; x < gradient(i).cols(); x++) {
        rms(i)(y, x) =
            beta2 * rms(i)(y, x) + (1.0f - beta2) * gradient(i)(y, x) * gradient(i)(y, x);
      }
    }
  }
}
