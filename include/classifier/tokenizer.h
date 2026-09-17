#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace classifier {

  struct TokenizerOptions {
    bool lowercase{ true };
    bool split_camel_case{ true };
    bool preserve_operators{ true };
    size_t ngram_min{ 1 };
    size_t ngram_max{ 1 }; // Set to 2 for Bigrams, 3 for Trigrams
  };

  class Tokenizer {
  public:
    explicit Tokenizer(TokenizerOptions options = {}) : options_(options) {}

    [[nodiscard]] std::vector<std::string> tokenize(std::string_view text) const;

  private:
    TokenizerOptions options_;

    void split_and_append_subwords(std::string_view token, std::vector<std::string> &out) const;
  };

} // namespace classifier