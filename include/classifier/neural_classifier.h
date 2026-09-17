#pragma once

#include "classifier/types.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace classifier {

  struct NeuralTrainConfig {
    float learning_rate{ 0.05f };
    float l2_reg{ 1e-5f };
    size_t hidden_dim{ 32 };
    size_t epochs{ 100 };
    size_t batch_size{ 32 };
    uint64_t random_seed{ 42 };
    bool verbose{ false };
  };

  class NeuralClassifier {
  public:
    NeuralClassifier() = default;
    NeuralClassifier(size_t num_features, size_t hidden_dim = 32, size_t num_classes = 3);

    void init(size_t num_features, size_t hidden_dim = 32, size_t num_classes = 3, uint64_t seed = 42);

    // Forward pass
    void forward(
      const SparseVector &x,
      std::vector<float> &hidden,
      std::array<float, 3> &logits
    ) const;

    [[nodiscard]] std::array<float, 3> predict_proba(const SparseVector &x) const;
    [[nodiscard]] Tier predict(const SparseVector &x) const;

    // Training via mini-batch SGD + Backpropagation
    void train(
      const std::vector<SparseVector> &X,
      const std::vector<Tier> &y,
      const NeuralTrainConfig &config
    );

    [[nodiscard]] float compute_loss(
      const std::vector<SparseVector> &X,
      const std::vector<Tier> &y,
      float l2_reg = 0.0f
    ) const;

    [[nodiscard]] size_t num_features() const noexcept { return num_features_; }
    [[nodiscard]] size_t hidden_dim() const noexcept { return hidden_dim_; }

    void save(std::ostream &os) const;
    void load(std::istream &is);

  private:
    size_t num_features_{ 0 };
    size_t hidden_dim_{ 32 };
    size_t num_classes_{ 3 };

    // Parameters:
    // W1: [hidden_dim_ * num_features_]
    // b1: [hidden_dim_]
    // W2: [num_classes_ * hidden_dim_]
    // b2: [num_classes_]
    std::vector<float> W1_;
    std::vector<float> b1_;
    std::vector<float> W2_;
    std::array<float, 3> b2_{ 0.0f, 0.0f, 0.0f };

    static std::array<float, 3> softmax(const std::array<float, 3> &logits);
  };

} // namespace classifier