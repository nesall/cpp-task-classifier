#pragma once

#include "classifier/types.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace classifier {

  struct TrainConfig {
    float learning_rate{ 0.1f };
    float l2_reg{ 1e-4f };
    size_t epochs{ 50 };
    size_t batch_size{ 32 };
    uint64_t random_seed{ 42 };
    bool verbose{ false };
  };

  class LogisticRegression {
  public:
    LogisticRegression() = default;
    LogisticRegression(size_t num_features, size_t num_classes = 3);

    void init(size_t num_features, size_t num_classes = 3);

    // Compute raw logits z = W * x + b
    void compute_logits(const SparseVector &x, std::array<float, 3> &logits) const;

    // Numerically stable softmax: logits -> probabilities
    [[nodiscard]] static std::array<float, 3> softmax(const std::array<float, 3> &logits);

    [[nodiscard]] std::array<float, 3> predict_proba(const SparseVector &x) const;
    [[nodiscard]] Tier predict(const SparseVector &x) const;

    [[nodiscard]] std::array<float, 3> predict_raw_scores(const SparseVector &x) const;

    // Training entrypoint
    void train(
      const std::vector<SparseVector> &X,
      const std::vector<Tier> &y,
      const TrainConfig &config
    );

    // Evaluate cross-entropy loss over a dataset
    [[nodiscard]] float compute_loss(
      const std::vector<SparseVector> &X,
      const std::vector<Tier> &y,
      float l2_reg = 0.0f
    ) const;

    // Direct parameter inspection (for tests & feature weight analysis)
    [[nodiscard]] float get_weight(size_t class_idx, size_t feature_idx) const;
    void set_weight(size_t class_idx, size_t feature_idx, float val);
    [[nodiscard]] float get_bias(size_t class_idx) const;
    void set_bias(size_t class_idx, float val);

    [[nodiscard]] size_t num_features() const noexcept { return num_features_; }
    [[nodiscard]] size_t num_classes() const noexcept { return num_classes_; }

    // Serialization
    void save(std::ostream &os) const;
    void load(std::istream &is);

  private:
    size_t num_features_{ 0 };
    size_t num_classes_{ 3 };
    // Flattened weights: index = class_idx * num_features_ + feature_idx
    std::vector<float> weights_;
    std::array<float, 3> biases_{ 0.0f, 0.0f, 0.0f };
  };

} // namespace classifier
