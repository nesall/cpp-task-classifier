#pragma once

#include "classifier/logistic_regression.h"
#include "classifier/tfidf.h"
#include "classifier/tokenizer.h"
#include "classifier/types.h"
#include "classifier/vocabulary.h"
#include "classifier/neural_classifier.h"
#include <memory>
#include <string_view>

namespace classifier {

  class IClassifier {
  public:
    virtual ~IClassifier() = default;
    [[nodiscard]] virtual ClassificationResult classify(std::string_view text, Tier sesson_ctx = Tier::Unknown) const = 0;
  };

  struct ClassificationThresholds {
    float tier3_threshold{ 0.30f };
    float tier2_threshold{ 0.35f };
    float tier3_margin{ 0.05f };
    float tier2_margin{ 0.05f };
    float min_t3_prob{ 0.30f };
    float min_t2_prob{ 0.30f };
  };

  enum class ModelType : uint8_t {
    Linear = 0,
    ResidualMLP = 1
  };

  struct ResidualConfig {
    size_t hidden_dim{ 16 };       // Compact bottleneck for non-linear residual corrections
    float learning_rate{ 0.15f };
    float l2_reg{ 1e-5f };
    size_t epochs{ 100 };
    size_t batch_size{ 32 };
  };

  class EmbeddedClassifier final : public IClassifier {
  public:
    explicit EmbeddedClassifier(
      ModelType model_type = ModelType::Linear,
      TokenizerOptions tokenizer_opts = {},
      ClassificationThresholds thresholds = {}
    );
    ~EmbeddedClassifier() override = default;

    void set_model_type(ModelType type) noexcept { model_type_ = type; }
    [[nodiscard]] ModelType model_type() const noexcept { return model_type_; }

    void set_tokenizer_options(TokenizerOptions opts);
    [[nodiscard]] const TokenizerOptions &tokenizer_options() const noexcept { return tokenizer_options_; }

    void set_thresholds(ClassificationThresholds thresholds) noexcept { thresholds_ = thresholds; }
    [[nodiscard]] const ClassificationThresholds &thresholds() const noexcept { return thresholds_; }

    [[nodiscard]] ClassificationResult classify(std::string_view text, Tier sesson_ctx = Tier::Unknown) const override;

    void fit(
      const std::vector<std::string> &texts,
      const std::vector<Tier> &labels,
      size_t max_vocab_size = 15000,
      uint32_t min_df = 2,
      const TrainConfig &linear_config = {},
      const ResidualConfig &residual_config = {}
    );

    void save(std::ostream &os) const;
    void load(std::istream &is);

    bool save_to_file(std::string_view filepath) const;
    bool load_from_file(std::string_view filepath);

    [[nodiscard]] const Vocabulary &vocabulary() const noexcept { return vocab_; }
    [[nodiscard]] const LogisticRegression &linear_model() const noexcept { return lr_; }
    [[nodiscard]] const NeuralClassifier &residual_model() const noexcept { return mlp_; }

  private:
    ModelType model_type_{ ModelType::Linear };
    TokenizerOptions tokenizer_options_{};
    ClassificationThresholds thresholds_{};
    std::unique_ptr<Tokenizer> tokenizer_;
    Vocabulary vocab_;
    TfidfVectorizer tfidf_;
    LogisticRegression lr_;
    NeuralClassifier mlp_;

    static constexpr uint32_t EMBEDDED_MAGIC = 0x454D4232; // 'EMB2'
  };

} // namespace classifier
