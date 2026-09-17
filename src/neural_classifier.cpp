#include "classifier/neural_classifier.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace classifier {

  static constexpr uint32_t MLP_MAGIC = 0x4E4E4D31; // 'NNM1'

  NeuralClassifier::NeuralClassifier(size_t num_features, size_t hidden_dim, size_t num_classes) {
    init(num_features, hidden_dim, num_classes);
  }

  void NeuralClassifier::init(size_t num_features, size_t hidden_dim, size_t num_classes, uint64_t seed) {
    num_features_ = num_features;
    hidden_dim_ = hidden_dim;
    num_classes_ = num_classes;

    W1_.assign(hidden_dim_ * num_features_, 0.0f);
    b1_.assign(hidden_dim_, 0.0f);
    W2_.assign(num_classes_ * hidden_dim_, 0.0f);
    b2_.fill(0.0f);

    std::mt19937_64 rng(seed);

    // He initialization for ReLU: scale = sqrt(2.0 / fan_in)
    float scale1 = std::sqrt(2.0f / static_cast<float>(std::max<size_t>(1, num_features_)));
    std::uniform_real_distribution<float> dist1(-scale1, scale1);
    for (auto &w : W1_) w = dist1(rng);

    float scale2 = std::sqrt(2.0f / static_cast<float>(hidden_dim_));
    std::uniform_real_distribution<float> dist2(-scale2, scale2);
    for (auto &w : W2_) w = dist2(rng);
  }

  void NeuralClassifier::forward(
    const SparseVector &x,
    std::vector<float> &hidden,
    std::array<float, 3> &logits
  ) const {
    hidden.assign(b1_.begin(), b1_.end());

    // Sparse GEMV: hidden = ReLU(W1 * x + b1)
    for (const auto &feat : x) {
      if (feat.index < num_features_) {
        const size_t col = feat.index;
        const float val = feat.value;
        for (size_t h = 0; h < hidden_dim_; ++h) {
          hidden[h] += W1_[h * num_features_ + col] * val;
        }
      }
    }

    // ReLU activation
    for (size_t h = 0; h < hidden_dim_; ++h) {
      hidden[h] = std::max(0.0f, hidden[h]);
    }

    // Dense GEMV: logits = W2 * hidden + b2
    logits = b2_;
    for (size_t k = 0; k < num_classes_; ++k) {
      const size_t offset = k * hidden_dim_;
      float dot = 0.0f;
      for (size_t h = 0; h < hidden_dim_; ++h) {
        dot += W2_[offset + h] * hidden[h];
      }
      logits[k] += dot;
    }
  }

  std::array<float, 3> NeuralClassifier::softmax(const std::array<float, 3> &logits) {
    float m = std::max({ logits[0], logits[1], logits[2] });
    std::array<float, 3> exp_v{
        std::exp(logits[0] - m),
        std::exp(logits[1] - m),
        std::exp(logits[2] - m)
    };
    float s = exp_v[0] + exp_v[1] + exp_v[2];
    float inv_s = 1.0f / s;
    return { exp_v[0] * inv_s, exp_v[1] * inv_s, exp_v[2] * inv_s };
  }

  std::array<float, 3> NeuralClassifier::predict_proba(const SparseVector &x) const {
    std::vector<float> h(hidden_dim_);
    std::array<float, 3> z;
    forward(x, h, z);
    return softmax(z);
  }

  Tier NeuralClassifier::predict(const SparseVector &x) const {
    auto p = predict_proba(x);
    size_t best = 0;
    float mx = p[0];
    for (size_t k = 1; k < 3; ++k) {
      if (p[k] > mx) {
        mx = p[k];
        best = k;
      }
    }
    return static_cast<Tier>(best);
  }

  float NeuralClassifier::compute_loss(
    const std::vector<SparseVector> &X,
    const std::vector<Tier> &y,
    float l2_reg
  ) const {
    if (X.empty()) return 0.0f;
    double ce = 0.0;
    std::vector<float> h(hidden_dim_);
    std::array<float, 3> z;

    for (size_t i = 0; i < X.size(); ++i) {
      forward(X[i], h, z);
      float m = std::max({ z[0], z[1], z[2] });
      double sum_e = std::exp(z[0] - m) + std::exp(z[1] - m) + std::exp(z[2] - m);
      double lse = m + std::log(sum_e);
      auto label = static_cast<size_t>(y[i]);
      if (label < 3) {
        ce += (lse - static_cast<double>(z[label]));
      }
    }

    double total = ce / static_cast<double>(X.size());
    if (l2_reg > 0.0f) {
      double w_norm = 0.0;
      for (float w : W1_) w_norm += w * w;
      for (float w : W2_) w_norm += w * w;
      total += 0.5 * static_cast<double>(l2_reg) * w_norm;
    }
    return static_cast<float>(total);
  }

  void NeuralClassifier::train(
    const std::vector<SparseVector> &X,
    const std::vector<Tier> &y,
    const NeuralTrainConfig &config
  ) {
    if (X.empty() || X.size() != y.size()) return;

    std::vector<size_t> idxs(X.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::mt19937_64 rng(config.random_seed);

    // Forward activations
    std::vector<float> h(hidden_dim_);
    std::array<float, 3> z;

    // Gradient accumulators
    std::vector<float> grad_W1(W1_.size(), 0.0f);
    std::vector<float> grad_b1(b1_.size(), 0.0f);
    std::vector<float> grad_W2(W2_.size(), 0.0f);
    std::array<float, 3> grad_b2{ 0.0f, 0.0f, 0.0f };

    std::vector<float> delta_h(hidden_dim_, 0.0f);
    std::vector<uint32_t> touched_cols;

    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
      std::shuffle(idxs.begin(), idxs.end(), rng);
      float lr_epoch = config.learning_rate / (1.0f + 0.012f * static_cast<float>(epoch));

      for (size_t b_start = 0; b_start < X.size(); b_start += config.batch_size) {
        size_t b_end = std::min(b_start + config.batch_size, X.size());
        size_t b_sz = b_end - b_start;
        float inv_b = 1.0f / static_cast<float>(b_sz);

        std::fill(grad_b1.begin(), grad_b1.end(), 0.0f);
        std::fill(grad_W2.begin(), grad_W2.end(), 0.0f);
        grad_b2.fill(0.0f);
        touched_cols.clear();

        for (size_t b = b_start; b < b_end; ++b) {
          size_t s = idxs[b];
          const auto &x = X[s];
          auto target = static_cast<size_t>(y[s]);

          forward(x, h, z);
          auto p = softmax(z);

          // Output layer error: e_k = p_k - y_k
          std::array<float, 3> e = p;
          if (target < 3) e[target] -= 1.0f;

          for (size_t k = 0; k < 3; ++k) {
            grad_b2[k] += e[k];
            const size_t off = k * hidden_dim_;
            for (size_t d = 0; d < hidden_dim_; ++d) {
              grad_W2[off + d] += e[k] * h[d];
            }
          }

          // Backprop into hidden layer
          for (size_t d = 0; d < hidden_dim_; ++d) {
            float sum_err = 0.0f;
            for (size_t k = 0; k < 3; ++k) {
              sum_err += e[k] * W2_[k * hidden_dim_ + d];
            }
            // Derivative of ReLU
            delta_h[d] = (h[d] > 0.0f) ? sum_err : 0.0f;
            grad_b1[d] += delta_h[d];
          }

          // Backprop into sparse W1
          for (const auto &feat : x) {
            if (feat.index < num_features_) {
              const size_t col = feat.index;
              const float val = feat.value;
              for (size_t d = 0; d < hidden_dim_; ++d) {
                grad_W1[d * num_features_ + col] += delta_h[d] * val;
              }
              touched_cols.push_back(static_cast<uint32_t>(col));
            }
          }
        }

        // Apply updates to Layer 2
        float decay = 1.0f - (lr_epoch * config.l2_reg);
        for (size_t k = 0; k < 3; ++k) {
          b2_[k] -= lr_epoch * (grad_b2[k] * inv_b);
          const size_t off = k * hidden_dim_;
          for (size_t d = 0; d < hidden_dim_; ++d) {
            W2_[off + d] = (W2_[off + d] * decay) - (lr_epoch * grad_W2[off + d] * inv_b);
          }
        }

        // Apply updates to Layer 1 biases
        for (size_t d = 0; d < hidden_dim_; ++d) {
          b1_[d] -= lr_epoch * (grad_b1[d] * inv_b);
        }

        // Apply updates to sparse Layer 1 weights
        std::sort(touched_cols.begin(), touched_cols.end());
        touched_cols.erase(std::unique(touched_cols.begin(), touched_cols.end()), touched_cols.end());

        for (uint32_t col : touched_cols) {
          for (size_t d = 0; d < hidden_dim_; ++d) {
            const size_t idx = d * num_features_ + col;
            W1_[idx] = (W1_[idx] * decay) - (lr_epoch * grad_W1[idx] * inv_b);
            grad_W1[idx] = 0.0f; // Reset accumulator
          }
        }
      }

      if (config.verbose && (epoch + 1) % 15 == 0) {
        std::cout << "Neural Epoch " << (epoch + 1) << "/" << config.epochs
          << " - Loss: " << compute_loss(X, y, config.l2_reg) << "\n";
      }
    }
  }

  void NeuralClassifier::save(std::ostream &os) const {
    os.write(reinterpret_cast<const char *>(&MLP_MAGIC), sizeof(MLP_MAGIC));
    uint64_t nf = num_features_;
    uint64_t hd = hidden_dim_;
    uint64_t nc = num_classes_;
    os.write(reinterpret_cast<const char *>(&nf), sizeof(nf));
    os.write(reinterpret_cast<const char *>(&hd), sizeof(hd));
    os.write(reinterpret_cast<const char *>(&nc), sizeof(nc));

    os.write(reinterpret_cast<const char *>(W1_.data()), sizeof(float) * W1_.size());
    os.write(reinterpret_cast<const char *>(b1_.data()), sizeof(float) * b1_.size());
    os.write(reinterpret_cast<const char *>(W2_.data()), sizeof(float) * W2_.size());
    os.write(reinterpret_cast<const char *>(b2_.data()), sizeof(float) * 3);
  }

  void NeuralClassifier::load(std::istream &is) {
    uint32_t magic = 0;
    is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (magic != MLP_MAGIC) throw std::runtime_error("Invalid MLP binary header");

    uint64_t nf = 0, hd = 0, nc = 0;
    is.read(reinterpret_cast<char *>(&nf), sizeof(nf));
    is.read(reinterpret_cast<char *>(&hd), sizeof(hd));
    is.read(reinterpret_cast<char *>(&nc), sizeof(nc));

    init(static_cast<size_t>(nf), static_cast<size_t>(hd), static_cast<size_t>(nc));
    is.read(reinterpret_cast<char *>(W1_.data()), sizeof(float) * W1_.size());
    is.read(reinterpret_cast<char *>(b1_.data()), sizeof(float) * b1_.size());
    is.read(reinterpret_cast<char *>(W2_.data()), sizeof(float) * W2_.size());
    is.read(reinterpret_cast<char *>(b2_.data()), sizeof(float) * 3);
  }

} // namespace classifier