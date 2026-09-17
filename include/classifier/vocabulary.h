#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace classifier {

  class Vocabulary {
  public:
    Vocabulary() = default;

    // Build vocabulary from pre-tokenized corpus
    void fit(
      const std::vector<std::vector<std::string>> &tokenized_corpus,
      size_t max_vocab_size = 10000,
      uint32_t min_df = 1
    );

    // Returns token index, or -1 if Out-Of-Vocabulary
    [[nodiscard]] int32_t get_index(std::string_view token) const;

    [[nodiscard]] std::string_view get_token(uint32_t index) const;

    [[nodiscard]] size_t size() const noexcept { return index_to_token_.size(); }

    [[nodiscard]] bool contains(std::string_view token) const;

    // Document frequencies per token index
    [[nodiscard]] const std::vector<uint32_t> &document_frequencies() const noexcept {
      return doc_frequencies_;
    }

    // Directly inject state (used for deserialization)
    void set_state(std::vector<std::string> tokens, std::vector<uint32_t> dfs);

  private:
    std::unordered_map<std::string, uint32_t> token_to_index_;
    std::vector<std::string> index_to_token_;
    std::vector<uint32_t> doc_frequencies_;
  };

} // namespace classifier
