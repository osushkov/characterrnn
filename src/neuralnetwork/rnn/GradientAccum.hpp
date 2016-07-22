#pragma once

#include "../../common/Common.hpp"
#include "../../common/Maybe.hpp"
#include "../../math/Math.hpp"
#include "Layer.hpp"
#include "RNNSpec.hpp"
#include <vector>

namespace neuralnetwork {
namespace rnn {

struct ConnectionAccum {
  EMatrix accumGradient;
  unsigned samples;

  ConnectionAccum(const EMatrix &gradient) : accumGradient(gradient), samples(1) {}

  EMatrix GetGradient(void) const {
    assert(samples > 0);
    return accumGradient * (1.0f / static_cast<float>(samples));
  }

  void AccumGradient(const EMatrix &gradient) {
    accumGradient += gradient;
    samples++;
  }
};

struct GradientAccum {
  vector<pair<LayerConnection, ConnectionAccum>> allWeightsAccum;

  void IncrementWeights(const LayerConnection &connection, const EMatrix &gradient) {
    for (auto &wa : allWeightsAccum) {
      if (wa.first == connection) {
        wa.second.AccumGradient(gradient);
        return;
      }
    }

    allWeightsAccum.emplace_back(connection, ConnectionAccum(gradient));
  }

  Maybe<EMatrix> GetGradient(const LayerConnection &connection) {
    for (auto &wa : allWeightsAccum) {
      if (wa.first == connection) {
        return Maybe<EMatrix>(wa.second.GetGradient());
      }
    }

    return Maybe<EMatrix>::none;
  }

  void DebugPrint(void) {
    cout << "num gradients accumulated: " << allWeightsAccum.size() << endl;
    for (const auto &wa : allWeightsAccum) {
      cout << "acc: " << wa.first.srcLayerId << " -> " << wa.first.dstLayerId <<
      " = " << wa.second.samples << endl;
    }
  }
};
}
}
