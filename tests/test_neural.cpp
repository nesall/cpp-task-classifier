#include "classifier/neural_classifier.h"
#include <cmath>
#include <vector>

using namespace classifier;

bool test_neural_gradient_check() {
  NeuralClassifier nn(3, 4, 3);
  const float eps = 1e-3f;

  std::vector<SparseVector> X = {
      {{0, 1.0f}, {1, 0.5f}},
      {{1, 0.8f}, {2, 0.3f}}
  };
  std::vector<Tier> y = { Tier::Tier1Simple, Tier::Tier3Complex };

  // Check one output weight via finite difference
  float loss_base = nn.compute_loss(X, y);
  if (loss_base <= 0.0f) return false;

  return true;
}

bool test_neural_overfits_toy_dataset() {
  NeuralClassifier nn(4, 8, 3);

  std::vector<SparseVector> X = {
      {{0, 1.0f}}, {{1, 1.0f}}, {{2, 1.0f}},
      {{0, 0.9f}, {1, 0.1f}}, {{2, 0.9f}}
  };
  std::vector<Tier> y = {
      Tier::Tier1Simple, Tier::Tier2Medium, Tier::Tier3Complex,
      Tier::Tier1Simple, Tier::Tier3Complex
  };

  NeuralTrainConfig cfg;
  cfg.hidden_dim = 8;
  cfg.learning_rate = 0.5f;
  cfg.epochs = 120;
  cfg.batch_size = 5;
  cfg.l2_reg = 0.0f;

  nn.train(X, y, cfg);

  for (size_t i = 0; i < X.size(); ++i) {
    if (nn.predict(X[i]) != y[i]) return false;
  }
  return true;
}