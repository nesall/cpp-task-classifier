#include "classifier/tfidf.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace classifier {

  void TfidfVectorizer::fit(const Vocabulary &vocab, size_t num_documents)
  {
    const auto &dfs = vocab.document_frequencies();
    idf_weights_.resize(vocab.size(), 0.0f);

    const double n = static_cast<double>(num_documents);

    for (size_t i = 0; i < vocab.size(); ++i) {
      const double df = static_cast<double>(dfs[i]);
      if (options_.smooth_idf) {
        // smooth_idf: ln((1 + N) / (1 + DF)) + 1.0
        idf_weights_[i] = static_cast<float>(std::log((1.0 + n) / (1.0 + df)) + 1.0);
      } else {
        // raw_idf: ln(N / DF) + 1.0
        idf_weights_[i] = static_cast<float>(std::log(n / std::max(1.0, df)) + 1.0);
      }
    }
  }

  SparseVector TfidfVectorizer::transform(
    const std::vector<std::string> &tokens,
    const Vocabulary &vocab
  ) const
  {
    if (tokens.empty() || vocab.size() == 0 || idf_weights_.empty()) {
      return {};
    }

    // 1. Calculate raw term frequencies in this document
    std::unordered_map<uint32_t, uint32_t> tf_counts;
    for (const auto &token : tokens) {
      int32_t idx = vocab.get_index(token);
      if (idx >= 0) {
        tf_counts[static_cast<uint32_t>(idx)]++;
      }
    }

    if (tf_counts.empty()) {
      return {}; // All tokens are OOV
    }

    // 2. Compute TF * IDF weights
    SparseVector vec;
    vec.reserve(tf_counts.size());
    double sum_sq = 0.0;

    for (const auto &[idx, count] : tf_counts) {
      double tf = options_.sublinear_tf
        ? (1.0 + std::log(static_cast<double>(count)))
        : static_cast<double>(count);

      double val = tf * static_cast<double>(idf_weights_[idx]);
      sum_sq += val * val;
      vec.push_back({ idx, static_cast<float>(val) });
    }

    // 3. Apply L2 normalization
    if (options_.l2_normalize && sum_sq > 0.0) {
      float inv_norm = static_cast<float>(1.0 / std::sqrt(sum_sq));
      for (auto &feat : vec) {
        feat.value *= inv_norm;
      }
    }

    // 4. Sort by feature index (ensures deterministic dot products)
    std::sort(vec.begin(), vec.end(), [](const SparseFeature &a, const SparseFeature &b) {
      return a.index < b.index;
      });

    return vec;
  }

} // namespace classifier
