#include "classifier/vocabulary.h"
#include "classifier/tfidf.h"
#include <cmath>
#include <vector>
#include <string>

using namespace classifier;

static bool approx_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) < eps;
}

bool test_vocabulary_pruning_and_order() {
  std::vector<std::vector<std::string>> corpus = {
      {"rare", "common", "common"},
      {"common", "medium"},
      {"common", "medium"}
  };

  Vocabulary vocab;
  // Set min_df = 2 -> "rare" should be pruned because its df is 1
  vocab.fit(corpus, 10, 2);

  if (vocab.size() != 2) return false;
  if (vocab.contains("rare")) return false;
  if (!vocab.contains("common")) return false;
  if (!vocab.contains("medium")) return false;

  // "common" has higher TF (4) than "medium" (2), so index 0 must be "common"
  if (vocab.get_index("common") != 0) return false;
  if (vocab.get_index("medium") != 1) return false;

  return true;
}

bool test_tfidf_exact_weights_and_normalization() {
  std::vector<std::vector<std::string>> corpus = {
      {"vector", "vector", "int"},
      {"vector", "map"},
      {"map", "string"}
  };

  Vocabulary vocab;
  vocab.fit(corpus, 10, 1);

  TfidfVectorizer tfidf;
  tfidf.fit(vocab, corpus.size());

  // Check smoothed IDF for DF=2 (vector, map) and DF=1 (int, string)
  int32_t vector_idx = vocab.get_index("vector");
  int32_t int_idx = vocab.get_index("int");

  float expected_idf_df2 = static_cast<float>(std::log(4.0 / 3.0) + 1.0); // ~1.28768
  float expected_idf_df1 = static_cast<float>(std::log(4.0 / 2.0) + 1.0); // ~1.69315

  if (!approx_equal(tfidf.idf_weights()[vector_idx], expected_idf_df2)) return false;
  if (!approx_equal(tfidf.idf_weights()[int_idx], expected_idf_df1)) return false;

  // Transform Doc 0: {"vector", "vector", "int"}
  SparseVector vec = tfidf.transform({ "vector", "vector", "int" }, vocab);

  if (vec.size() != 2) return false;

  // Must be sorted by index
  if (vec[0].index >= vec[1].index) return false;

  // Verify L2 norm is 1.0
  float norm_sq = 0.0f;
  for (const auto &feat : vec) {
    norm_sq += feat.value * feat.value;
  }
  if (!approx_equal(std::sqrt(norm_sq), 1.0f)) return false;

  // Verify sublinear TF effect: count=2 for vector
  double sublinear_tf = 1.0 + std::log(2.0);
  double unnorm_vector = sublinear_tf * expected_idf_df2;
  double unnorm_int = 1.0 * expected_idf_df1;
  double unnorm_norm = std::sqrt(unnorm_vector * unnorm_vector + unnorm_int * unnorm_int);

  float expected_vector_val = static_cast<float>(unnorm_vector / unnorm_norm);
  float expected_int_val = static_cast<float>(unnorm_int / unnorm_norm);

  for (const auto &feat : vec) {
    if (feat.index == static_cast<uint32_t>(vector_idx)) {
      if (!approx_equal(feat.value, expected_vector_val)) return false;
    } else if (feat.index == static_cast<uint32_t>(int_idx)) {
      if (!approx_equal(feat.value, expected_int_val)) return false;
    }
  }

  return true;
}

bool test_tfidf_oov_tokens() {
  std::vector<std::vector<std::string>> corpus = {
      {"alpha", "beta"}
  };

  Vocabulary vocab;
  vocab.fit(corpus, 10, 1);

  TfidfVectorizer tfidf;
  tfidf.fit(vocab, corpus.size());

  // Completely unseen tokens
  SparseVector vec = tfidf.transform({ "gamma", "delta", "epsilon" }, vocab);
  if (!vec.empty()) return false;

  // Mixed seen and unseen tokens
  SparseVector mixed = tfidf.transform({ "alpha", "delta" }, vocab);
  if (mixed.size() != 1) return false;
  if (!approx_equal(mixed[0].value, 1.0f)) return false; // Single feature normalized = 1.0

  return true;
}
