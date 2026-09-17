#pragma once

#include "classifier/types.h"
#include "classifier/vocabulary.h"
#include <cstddef>
#include <string>
#include <vector>

namespace classifier {

  struct TfidfOptions {
    bool sublinear_tf{ true };
    bool smooth_idf{ true };
    bool l2_normalize{ true };
  };

  class TfidfVectorizer {
  public:
    explicit TfidfVectorizer(TfidfOptions options = {}) : options_(options) {}

    // Computes IDF weights from fitted vocabulary and corpus size N
    void fit(const Vocabulary &vocab, size_t num_documents);

    // Transforms a tokenized document into an L2-normalized SparseVector
    [[nodiscard]] SparseVector transform(const std::vector<std::string> &tokens, const Vocabulary &vocab) const;

    [[nodiscard]] const std::vector<float> &idf_weights() const noexcept {
      return idf_weights_;
    }
    
    // Directly inject state (used for deserialization)
    void set_idf_weights(std::vector<float> weights) {
      idf_weights_ = std::move(weights);
    }

  private:
    TfidfOptions options_;
    std::vector<float> idf_weights_;
  };

} // namespace classifier
