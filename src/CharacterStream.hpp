#pragma once

#include "common/Common.hpp"
#include "common/Maybe.hpp"
#include "math/OneHotVector.hpp"

#include <vector>

class CharacterStream {
public:
  CharacterStream(const string &filePath);
  ~CharacterStream();

  Maybe<math::OneHotVector> ReadCharacter(void);
  vector<math::OneHotVector> ReadCharacters(unsigned max);

private:
  struct CharacterStreamImpl;
  uptr<CharacterStreamImpl> impl;
};
