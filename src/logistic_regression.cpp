#include "classifier/logistic_regression.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace classifier {

  LogisticRegression::LogisticRegression(size_t num_features, size_t num_classes)
  {
    init(num_features, num_classes);
  }

  void LogisticRegression::init(size_t num_features, size_t num_classes)
  {
    if (num_classes != 3) {
      throw std::invalid_argument("Stage 3 classifier is specialized for 3 tiers.");
    }
    num_features_ = num_features;
    num_classes_ = num_classes;
    weights_.assign(num_classes_ * num_features_, 0.0f);
    biases_.fill(0.0f);
  }

  void LogisticRegression::compute_logits(const SparseVector &x, std::array<float, 3> &logits) const
  {
    logits = biases_;

    for (size_t k = 0; k < num_classes_; ++k) {
      const size_t class_offset = k * num_features_;
      float dot = 0.0f;
      for (const auto &feat : x) {
        if (feat.index < num_features_) {
          dot += weights_[class_offset + feat.index] * feat.value;
        }
      }
      logits[k] += dot;
    }
  }

  std::array<float, 3> LogisticRegression::softmax(const std::array<float, 3> &logits)
  {
    float max_logit = std::max({ logits[0], logits[1], logits[2] });
    std::array<float, 3> exp_vals{
        std::exp(logits[0] - max_logit),
        std::exp(logits[1] - max_logit),
        std::exp(logits[2] - max_logit)
    };

    float sum_exp = exp_vals[0] + exp_vals[1] + exp_vals[2];
    float inv_sum = 1.0f / sum_exp;

    return {
        exp_vals[0] * inv_sum,
        exp_vals[1] * inv_sum,
        exp_vals[2] * inv_sum
    };
  }

  std::array<float, 3> LogisticRegression::predict_proba(const SparseVector &x) const
  {
    std::array<float, 3> logits;
    compute_logits(x, logits);
    return softmax(logits);
  }

  std::array<float, 3> LogisticRegression::predict_raw_scores(const SparseVector &x) const
  {
    std::array<float, 3> scores = biases_;
    for (const auto &feat : x) {
      if (feat.index < num_features_) {
        for (size_t k = 0; k < num_classes_; ++k) {
          scores[k] += weights_[k * num_features_ + feat.index] * feat.value;
        }
      }
    }
    return scores;
  }

  Tier LogisticRegression::predict(const SparseVector &x) const
  {
    auto proba = predict_proba(x);
    size_t best_idx = 0;
    float best_p = proba[0];

    for (size_t k = 1; k < 3; ++k) {
      if (proba[k] > best_p) {
        best_p = proba[k];
        best_idx = k;
      }
    }
    return static_cast<Tier>(best_idx);
  }

  float LogisticRegression::compute_loss(const std::vector<SparseVector> &X, const std::vector<Tier> &y, float l2_reg, std::array<float, 3> clsw) const
  {
    if (X.empty() || X.size() != y.size()) {
      return 0.0f;
    }

    double total_ce = 0.0;
    std::array<float, 3> logits;

    for (size_t i = 0; i < X.size(); ++i) {
      compute_logits(X[i], logits);
      float m = std::max({ logits[0], logits[1], logits[2] });
      double sum_exp = std::exp(logits[0] - m) + std::exp(logits[1] - m) + std::exp(logits[2] - m);
      double log_sum_exp = m + std::log(sum_exp);

      auto target_k = static_cast<size_t>(y[i]);
      if (target_k < 3) {
        total_ce += clsw[target_k] * (log_sum_exp - static_cast<double>(logits[target_k]));
      }
    }

    double loss = total_ce / static_cast<double>(X.size());

    if (l2_reg > 0.0f) {
      double w_sq = 0.0;
      for (float w : weights_) {
        w_sq += static_cast<double>(w) * static_cast<double>(w);
      }
      loss += 0.5 * static_cast<double>(l2_reg) * w_sq;
    }

    return static_cast<float>(loss);
  }

  void LogisticRegression::train(const std::vector<SparseVector> &X, const std::vector<Tier> &y, const TrainConfig &config)
  {
    if (X.empty() || X.size() != y.size()) return;

    const size_t n_samples = X.size();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937_64 rng(config.random_seed);

    std::vector<float> grad_w(num_classes_ * num_features_, 0.0f);
    std::array<float, 3> grad_b{ 0.0f, 0.0f, 0.0f };
    std::vector<uint32_t> touched_features;

    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
      std::shuffle(indices.begin(), indices.end(), rng);

      // At the top of the epoch loop:
      float lr_epoch = config.learning_rate / (1.0f + 0.015f * static_cast<float>(epoch));

      for (size_t batch_start = 0; batch_start < n_samples; batch_start += config.batch_size) {
        size_t batch_end = std::min(batch_start + config.batch_size, n_samples);
        size_t current_batch_size = batch_end - batch_start;
        float inv_batch = 1.0f / static_cast<float>(current_batch_size);

        grad_b.fill(0.0f);
        touched_features.clear();

        // Accumulate mini-batch gradients
        for (size_t i = batch_start; i < batch_end; ++i) {
          size_t sample_idx = indices[i];
          const auto &x = X[sample_idx];
          auto target_k = static_cast<size_t>(y[sample_idx]);

          std::array<float, 3> probs = predict_proba(x);

          // Errors: e_k = prob_k - 1 (for target) else prob_k
          std::array<float, 3> errors = probs;
          if (target_k < 3) {
            errors[target_k] -= 1.0f;
          }
          float w = config.class_weights[target_k < 3 ? target_k : 0];
          for (size_t k = 0; k < 3; ++k) errors[k] *= w;

          // Bias gradient
          grad_b[0] += errors[0];
          grad_b[1] += errors[1];
          grad_b[2] += errors[2];

          // Sparse weight gradient accumulation
          for (const auto &feat : x) {
            if (feat.index >= num_features_) continue;

            for (size_t k = 0; k < 3; ++k) {
              size_t w_idx = k * num_features_ + feat.index;
              grad_w[w_idx] += errors[k] * feat.value;
            }
            touched_features.push_back(feat.index);
          }
        }

        // Update biases
        for (size_t k = 0; k < 3; ++k) {
          biases_[k] -= lr_epoch * (grad_b[k] * inv_batch);
        }

        // Update weights (decoupled L2 weight decay + sparse gradient step)
        float decay_factor = 1.0f - (lr_epoch * config.l2_reg);

        // Deduplicate touched feature list for sparse updates
        std::sort(touched_features.begin(), touched_features.end());
        touched_features.erase(
          std::unique(touched_features.begin(), touched_features.end()),
          touched_features.end()
        );

        for (uint32_t f_idx : touched_features) {
          for (size_t k = 0; k < 3; ++k) {
            size_t w_idx = k * num_features_ + f_idx;
            float g = grad_w[w_idx] * inv_batch;
            weights_[w_idx] = (weights_[w_idx] * decay_factor) - (lr_epoch * g);
            grad_w[w_idx] = 0.0f; // Reset buffer
          }
        }
      }

      if (config.verbose && (epoch + 1) % 10 == 0) {
        float loss = compute_loss(X, y, config.l2_reg, config.class_weights);
        std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
          << " - Loss: " << loss << "\n";
      }
    }
  }

  float LogisticRegression::get_weight(size_t class_idx, size_t feature_idx) const
  {
    return weights_[class_idx * num_features_ + feature_idx];
  }

  void LogisticRegression::set_weight(size_t class_idx, size_t feature_idx, float val)
  {
    weights_[class_idx * num_features_ + feature_idx] = val;
  }

  float LogisticRegression::get_bias(size_t class_idx) const
  {
    return biases_[class_idx];
  }

  void LogisticRegression::set_bias(size_t class_idx, float val)
  {
    biases_[class_idx] = val;
  }

  // Binary serialization: [magic(4B)][num_features(8B)][num_classes(8B)][biases(12B)][weights(num_classes * num_features * 4B)]
  static constexpr uint32_t MODEL_MAGIC = 0x4D4C5231; // 'MLR1'

  void LogisticRegression::save(std::ostream &os) const
  {
    os.write(reinterpret_cast<const char *>(&MODEL_MAGIC), sizeof(MODEL_MAGIC));
    uint64_t nf = num_features_;
    uint64_t nc = num_classes_;
    os.write(reinterpret_cast<const char *>(&nf), sizeof(nf));
    os.write(reinterpret_cast<const char *>(&nc), sizeof(nc));
    os.write(reinterpret_cast<const char *>(biases_.data()), sizeof(float) * 3);
    os.write(reinterpret_cast<const char *>(weights_.data()), sizeof(float) * weights_.size());
  }

  void LogisticRegression::load(std::istream &is)
  {
    uint32_t magic = 0;
    is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (magic != MODEL_MAGIC) {
      throw std::runtime_error("Invalid model binary format or corrupted header.");
    }
    uint64_t nf = 0;
    uint64_t nc = 0;
    is.read(reinterpret_cast<char *>(&nf), sizeof(nf));
    is.read(reinterpret_cast<char *>(&nc), sizeof(nc));

    init(static_cast<size_t>(nf), static_cast<size_t>(nc));

    is.read(reinterpret_cast<char *>(biases_.data()), sizeof(float) * 3);
    is.read(reinterpret_cast<char *>(weights_.data()), sizeof(float) * weights_.size());
  }

} // namespace classifier
