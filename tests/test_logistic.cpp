#include "classifier/logistic_regression.h"
#include <cmath>
#include <sstream>
#include <vector>

using namespace classifier;

static bool approx_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) < eps;
}

bool test_softmax_numerical_stability() {
  // Large logits that would overflow standard exp(1000)
  std::array<float, 3> extreme_logits{ 1000.0f, 1001.0f, 999.0f };
  auto probs = LogisticRegression::softmax(extreme_logits);

  float sum = probs[0] + probs[1] + probs[2];
  if (!approx_equal(sum, 1.0f, 1e-5f)) return false;

  // Largest logit (index 1) must have highest probability
  if (probs[1] <= probs[0] || probs[1] <= probs[2]) return false;

  // Must not be NaN or Inf
  for (float p : probs) {
    if (std::isnan(p) || std::isinf(p)) return false;
  }

  return true;
}

bool test_logistic_gradient_check() {
  const size_t n_features = 4;
  LogisticRegression model(n_features, 3);

  // Seed non-zero weights and biases
  model.set_weight(0, 0, 0.2f);  model.set_weight(0, 1, -0.3f);
  model.set_weight(1, 1, 0.5f);  model.set_weight(1, 2, -0.1f);
  model.set_weight(2, 2, 0.4f);  model.set_weight(2, 3, 0.7f);
  model.set_bias(0, 0.1f);
  model.set_bias(1, -0.2f);
  model.set_bias(2, 0.05f);

  std::vector<SparseVector> X = {
      {{0, 0.8f}, {1, 0.5f}},
      {{1, 0.3f}, {2, 0.9f}},
      {{2, 0.4f}, {3, 0.7f}}
  };
  std::vector<Tier> y = {
      Tier::Tier1Simple,
      Tier::Tier2Refactor,
      Tier::Tier3Complex
  };

  const float eps = 1e-3f;

  // Check analytical vs numerical gradient for weight W[1][2]
  size_t check_class = 1;
  size_t check_feat = 2;
  float original_w = model.get_weight(check_class, check_feat);

  model.set_weight(check_class, check_feat, original_w + eps);
  float loss_plus = model.compute_loss(X, y);

  model.set_weight(check_class, check_feat, original_w - eps);
  float loss_minus = model.compute_loss(X, y);

  model.set_weight(check_class, check_feat, original_w); // Restore

  float num_grad = (loss_plus - loss_minus) / (2.0f * eps);

  // Compute analytical gradient: (1/N) * sum_i (prob_1 - y_1) * x_feat
  float ana_grad = 0.0f;
  for (size_t i = 0; i < X.size(); ++i) {
    auto probs = model.predict_proba(X[i]);
    float error = probs[check_class] - (y[i] == static_cast<Tier>(check_class) ? 1.0f : 0.0f);
    for (const auto &feat : X[i]) {
      if (feat.index == check_feat) {
        ana_grad += error * feat.value;
      }
    }
  }
  ana_grad /= static_cast<float>(X.size());

  float rel_error = std::fabs(num_grad - ana_grad) / (std::fabs(num_grad) + std::fabs(ana_grad) + 1e-7f);
  return (rel_error < 1e-3f);
}

bool test_logistic_overfitting_toy_data() {
  const size_t n_features = 6;
  LogisticRegression model(n_features, 3);

  // 6 distinct samples cleanly separable by feature presence
  std::vector<SparseVector> X = {
      {{0, 1.0f}, {1, 0.5f}}, // Class 0
      {{0, 0.9f}, {1, 0.6f}}, // Class 0
      {{2, 1.0f}, {3, 0.7f}}, // Class 1
      {{2, 0.8f}, {3, 0.9f}}, // Class 1
      {{4, 1.0f}, {5, 0.8f}}, // Class 2
      {{4, 0.9f}, {5, 0.7f}}  // Class 2
  };
  std::vector<Tier> y = {
      Tier::Tier1Simple, Tier::Tier1Simple,
      Tier::Tier2Refactor, Tier::Tier2Refactor,
      Tier::Tier3Complex, Tier::Tier3Complex
  };

  TrainConfig config;
  config.learning_rate = 0.5f;
  config.l2_reg = 0.0f; // No regularization for strict overfit check
  config.epochs = 120;
  config.batch_size = 6;
  config.random_seed = 12345;

  model.train(X, y, config);

  // Must correctly predict every single training sample
  for (size_t i = 0; i < X.size(); ++i) {
    if (model.predict(X[i]) != y[i]) {
      return false;
    }
  }

  float final_loss = model.compute_loss(X, y);
  return (final_loss < 0.20f);
}

bool test_logistic_serialization_roundtrip() {
  LogisticRegression model(10, 3);
  model.set_weight(0, 1, 1.234f);
  model.set_weight(1, 5, -0.567f);
  model.set_weight(2, 9, 3.1415f);
  model.set_bias(0, 0.1f);
  model.set_bias(1, -0.2f);
  model.set_bias(2, 0.3f);

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  model.save(ss);

  LogisticRegression loaded;
  ss.seekg(0);
  loaded.load(ss);

  if (loaded.num_features() != 10) return false;
  if (loaded.num_classes() != 3) return false;

  if (!approx_equal(loaded.get_weight(0, 1), 1.234f)) return false;
  if (!approx_equal(loaded.get_weight(1, 5), -0.567f)) return false;
  if (!approx_equal(loaded.get_weight(2, 9), 3.1415f)) return false;
  if (!approx_equal(loaded.get_bias(0), 0.1f)) return false;
  if (!approx_equal(loaded.get_bias(1), -0.2f)) return false;
  if (!approx_equal(loaded.get_bias(2), 0.3f)) return false;

  return true;
}
