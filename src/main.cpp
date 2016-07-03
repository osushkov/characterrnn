
#include "CharacterStream.hpp"
#include "common/Common.hpp"

int main(int argc, char **argv) {
  srand(1234);
  cout << "hello world" << endl;

  string path(argv[1]);
  CharacterStream cstream(path);
  for (unsigned i = 0; i < 100; i++) {
    cstream.ReadCharacter();
  }

  vector<int> input{1, 2, 3, 4, 5};
  input = mapped_vector(input, [](int x) { return 2 * x; });

  for (const auto x : input) {
    cout << x << endl;
  }

  return 0;
}
