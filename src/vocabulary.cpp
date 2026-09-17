#include "classifier/vocabulary.h"
#include <algorithm>
#include <unordered_set>

namespace classifier {

  void Vocabulary::fit(
    const std::vector<std::vector<std::string>> &tokenized_corpus,
    size_t max_vocab_size,
    uint32_t min_df
  )
  {
    token_to_index_.clear();
    index_to_token_.clear();
    doc_frequencies_.clear();

    if (tokenized_corpus.empty()) {
      return;
    }

    // 1. Compute corpus-wide term frequency and document frequency
    std::unordered_map<std::string, uint32_t> tf_map;
    std::unordered_map<std::string, uint32_t> df_map;

    for (const auto &doc_tokens : tokenized_corpus) {
      std::unordered_set<std::string_view> unique_in_doc;
      for (const auto &token : doc_tokens) {
        tf_map[token]++;
        unique_in_doc.insert(token);
      }
      for (std::string_view token : unique_in_doc) {
        df_map[std::string(token)]++;
      }
    }

    // 2. Filter tokens by min_df
    struct TokenStats {
      std::string token;
      uint32_t tf;
      uint32_t df;
    };

    std::vector<TokenStats> candidates;
    candidates.reserve(df_map.size());

    for (const auto &[token, df] : df_map) {
      if (df >= min_df) {
        candidates.push_back({ token, tf_map[token], df });
      }
    }

    // 3. Sort by TF descending, then DF descending, then lexically (for determinism)
    std::sort(candidates.begin(), candidates.end(), [](const TokenStats &a, const TokenStats &b) {
      if (a.tf != b.tf) return a.tf > b.tf;
      if (a.df != b.df) return a.df > b.df;
      return a.token < b.token;
      });

    // 4. Truncate to max_vocab_size
    size_t target_size = std::min(candidates.size(), max_vocab_size);
    index_to_token_.reserve(target_size);
    doc_frequencies_.reserve(target_size);

    for (size_t i = 0; i < target_size; ++i) {
      const auto &item = candidates[i];
      token_to_index_[item.token] = static_cast<uint32_t>(i);
      index_to_token_.push_back(item.token);
      doc_frequencies_.push_back(item.df);
    }
  }

  int32_t Vocabulary::get_index(std::string_view token) const
  {
    auto it = token_to_index_.find(std::string(token));
    if (it != token_to_index_.end()) {
      return static_cast<int32_t>(it->second);
    }
    return -1;
  }

  std::string_view Vocabulary::get_token(uint32_t index) const
  {
    if (index < index_to_token_.size()) {
      return index_to_token_[index];
    }
    return {};
  }

  bool Vocabulary::contains(std::string_view token) const
  {
    return token_to_index_.contains(std::string(token));
  }

  void Vocabulary::set_state(std::vector<std::string> tokens, std::vector<uint32_t> dfs)
  {
    index_to_token_ = std::move(tokens);
    doc_frequencies_ = std::move(dfs);
    token_to_index_.clear();
    for (size_t i = 0; i < index_to_token_.size(); ++i) {
      token_to_index_[index_to_token_[i]] = static_cast<uint32_t>(i);
    }
  }

} // namespace classifier
