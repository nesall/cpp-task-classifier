#include "classifier/model.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace classifier {

  static constexpr uint32_t PIPELINE_MAGIC = 0x54494552; // 'TIER'
  static constexpr uint32_t PIPELINE_VERSION = 2;

  EmbeddedClassifier::EmbeddedClassifier(
    ModelType model_type,
    TokenizerOptions tokenizer_opts,
    ClassificationThresholds thresholds
  ) : model_type_(model_type),
    tokenizer_options_(tokenizer_opts),
    thresholds_(thresholds),
    tokenizer_(std::make_unique<Tokenizer>(tokenizer_opts)) {}

  void EmbeddedClassifier::set_tokenizer_options(TokenizerOptions opts)
  {
    tokenizer_options_ = opts;
    tokenizer_ = std::make_unique<Tokenizer>(opts);
  }

  ClassificationResult EmbeddedClassifier::classify(std::string_view text, Tier session_context) const
  {
    if (vocab_.size() == 0) {
      return { Tier::Unknown, {0.0f, 0.0f, 0.0f}, 0.0f };
    }

    auto tokens = tokenizer_->tokenize(text);

    if (tokens.empty()) {
      if (session_context != Tier::Unknown) {
        return ClassificationResult{
            .predicted_tier = session_context,
            .probabilities = {
                session_context == Tier::Tier1Simple ? 1.0f : 0.0f,
                session_context == Tier::Tier2Medium ? 1.0f : 0.0f,
                session_context == Tier::Tier3Complex ? 1.0f : 0.0f
            },
            .confidence = 1.0f
        };
      }
      return ClassificationResult{
          .predicted_tier = Tier::Tier1Simple,
          .probabilities = {1.0f, 0.0f, 0.0f},
          .confidence = 1.0f
      };
    }


    SparseVector x = tfidf_.transform(tokens, vocab_);

    // 1. Get raw scores from the linear model
    std::array<float, 3> logits = lr_.predict_raw_scores(x);

    //// 1b. Add MLP correction when ResidualMLP is selected
    // MLP makes it worse.
    //if (model_type_ == ModelType::ResidualMLP) {
    //  float alpha = 0.2f;
    //  std::vector<float> hidden;
    //  std::array<float, 3> mlp_logits;
    //  mlp_.forward(x, hidden, mlp_logits);
    //  for (size_t k = 0; k < 3; ++k) logits[k] += alpha * mlp_logits[k];
    //}

    // 2a. Standalone Query Length Prior
    float length_factor = std::clamp((static_cast<float>(tokens.size()) - 10.0f) / 40.0f, 0.0f, 1.0f);
    logits[1] += 0.4f * length_factor;
    logits[2] += 0.8f * length_factor;

    // 2b. High-Priority Concurrency & Structural Root Check
    // If explicit hard-boundary terms appear, suppress T1/T2 from overriding
    static const std::array<std::string_view, 6> hard_t3_stems = {
        "deadlock", "mutex", "thread-safe", "race condition", "atomic", "hazard pointer"
    };
    for (const auto &stem : hard_t3_stems) {
      if (text.find(stem) != std::string_view::npos) {
        logits[2] += 2.0f; // Direct priority nudge for undisputed concurrency keywords
        break;
      }
    }

    // 2c. Syntax Lookup Guard (std:: or syntax questions should not climb to Tier 2/3)
    if (text.find("std::") != std::string_view::npos &&
      (text.rfind("What does", 0) == 0 || text.rfind("How do I", 0) == 0)) {
      logits[0] += 1.2f; // Prioritize syntax lookup
    }

    // 2d. Session Context Inheritance (Decays as query gets longer)
    if (session_context != Tier::Unknown) {
      float context_weight = std::max(0.0f, 1.0f - (static_cast<float>(tokens.size()) / 15.0f));
      float boostFactor = 2.5f;
      // If ultra-short and continuing a session, the raw lexical signal is high-variance
      // conversational filler. Shrink raw logits toward zero before applying the context prior.
      const size_t ultraShortThreshold = 3;
      if (tokens.size() <= ultraShortThreshold) {
        float shrink_factor = 0.25f;
        for (float &l : logits) {
          l *= shrink_factor;
        }
        boostFactor += (ultraShortThreshold - tokens.size()) * 0.3f;
      }
      size_t target_idx = static_cast<size_t>(session_context);
      float context_boost = boostFactor * context_weight;
      logits[target_idx] += context_boost;
    }

    // 3. Standard Softmax
    float max_logit = std::max({ logits[0], logits[1], logits[2] });
    std::array<float, 3> exp_v{
        std::exp(logits[0] - max_logit),
        std::exp(logits[1] - max_logit),
        std::exp(logits[2] - max_logit)
    };
    float s = exp_v[0] + exp_v[1] + exp_v[2];
    float inv_s = 1.0f / s;
    std::array<float, 3> probs{ exp_v[0] * inv_s, exp_v[1] * inv_s, exp_v[2] * inv_s };

    // 4. Argmax baseline
    size_t best_idx = 0;
    float max_p = probs[0];
    for (size_t k = 1; k < 3; ++k) {
      if (probs[k] > max_p) {
        max_p = probs[k];
        best_idx = k;
      }
    }

    Tier chosen_tier = static_cast<Tier>(best_idx);

    // 5. Relative Margin Guardrails
    if (best_idx != 2 && (max_p - probs[2]) <= thresholds_.tier3_margin && probs[2] >= thresholds_.min_t3_prob) {
      chosen_tier = Tier::Tier3Complex;
    } else if (best_idx == 0 && (max_p - probs[1]) <= thresholds_.tier2_margin && probs[1] >= thresholds_.min_t2_prob) {
      chosen_tier = Tier::Tier2Medium;
    }

    return { chosen_tier, probs, max_p };
  }

  void EmbeddedClassifier::fit(
    const std::vector<std::string> &texts,
    const std::vector<Tier> &labels,
    size_t max_vocab_size,
    uint32_t min_df,
    const TrainConfig &linear_config,
    const ResidualConfig &residual_config
  )
  {
    if (texts.empty() || texts.size() != labels.size()) return;

    // 1. Build vocabulary and tokenized corpus
    std::vector<std::vector<std::string>> tokenized_corpus;
    tokenized_corpus.reserve(texts.size());
    for (const auto &text : texts) {
      tokenized_corpus.push_back(tokenizer_->tokenize(text));
    }

    vocab_.fit(tokenized_corpus, max_vocab_size, min_df);
    tfidf_.fit(vocab_, texts.size());

    // 2. Transform to sparse features
    std::vector<SparseVector> X;
    X.reserve(texts.size());
    for (const auto &doc : tokenized_corpus) {
      X.push_back(tfidf_.transform(doc, vocab_));
    }

    // 3. Fit the Base Linear Model
    lr_.init(vocab_.size(), 3);
    lr_.train(X, labels, linear_config);

    // 4. If ResidualMLP selected, train residual network on top
    if (model_type_ == ModelType::ResidualMLP) {
      NeuralTrainConfig nn_cfg;
      nn_cfg.hidden_dim = residual_config.hidden_dim;
      nn_cfg.learning_rate = residual_config.learning_rate;
      nn_cfg.l2_reg = residual_config.l2_reg;
      nn_cfg.epochs = residual_config.epochs;
      nn_cfg.batch_size = residual_config.batch_size;
      nn_cfg.random_seed = linear_config.random_seed;
      nn_cfg.verbose = linear_config.verbose;

      mlp_.init(vocab_.size(), residual_config.hidden_dim, 3, linear_config.random_seed);
      mlp_.train(X, labels, nn_cfg);
    }
  }

  void EmbeddedClassifier::save(std::ostream &os) const
  {
    os.write(reinterpret_cast<const char *>(&PIPELINE_MAGIC), sizeof(PIPELINE_MAGIC));
    os.write(reinterpret_cast<const char *>(&PIPELINE_VERSION), sizeof(PIPELINE_VERSION));

    // Save vocabulary
    uint64_t vocab_size = vocab_.size();
    os.write(reinterpret_cast<const char *>(&vocab_size), sizeof(vocab_size));
    for (size_t i = 0; i < vocab_size; ++i) {
      std::string_view token = vocab_.get_token(static_cast<uint32_t>(i));
      uint32_t len = static_cast<uint32_t>(token.size());
      os.write(reinterpret_cast<const char *>(&len), sizeof(len));
      os.write(token.data(), len);
      uint32_t df = vocab_.document_frequencies()[i];
      os.write(reinterpret_cast<const char *>(&df), sizeof(df));
    }

    // Save IDF weights
    const auto &idf = tfidf_.idf_weights();
    uint64_t idf_size = idf.size();
    os.write(reinterpret_cast<const char *>(&idf_size), sizeof(idf_size));
    os.write(reinterpret_cast<const char *>(idf.data()), sizeof(float) * idf.size());

    // Save tokenizer_options_
    uint32_t ngram_min = static_cast<uint32_t>(tokenizer_options_.ngram_min);
    uint32_t ngram_max = static_cast<uint32_t>(tokenizer_options_.ngram_max);
    os.write(reinterpret_cast<const char *>(&ngram_min), sizeof(ngram_min));
    os.write(reinterpret_cast<const char *>(&ngram_max), sizeof(ngram_max));
    uint8_t lowercase = tokenizer_options_.lowercase ? 1 : 0;
    uint8_t split_camel_case = tokenizer_options_.split_camel_case ? 1 : 0;
    uint8_t preserve_operators = tokenizer_options_.preserve_operators ? 1 : 0;
    os.write(reinterpret_cast<const char *>(&lowercase), sizeof(lowercase));
    os.write(reinterpret_cast<const char *>(&split_camel_case), sizeof(split_camel_case));
    os.write(reinterpret_cast<const char *>(&preserve_operators), sizeof(preserve_operators));


    // Save Logistic Regression weights
    lr_.save(os);

    auto mt = static_cast<uint8_t>(model_type_);
    os.write(reinterpret_cast<const char *>(&mt), sizeof(mt));
    if (model_type_ == ModelType::ResidualMLP) {
      mlp_.save(os);
    }

    // Save classification thresholds (POD struct of 6 floats, no padding)
    os.write(reinterpret_cast<const char *>(&thresholds_), sizeof(ClassificationThresholds));
  }

  void EmbeddedClassifier::load(std::istream &is)
  {
    uint32_t magic = 0;
    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char *>(&version), sizeof(version));

    if (magic != PIPELINE_MAGIC || version != PIPELINE_VERSION) {
      throw std::runtime_error("Corrupted or incompatible model binary.");
    }

    // Load vocabulary
    uint64_t vocab_size = 0;
    is.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));

    std::vector<std::string> loaded_tokens;
    std::vector<uint32_t> loaded_dfs;
    loaded_tokens.reserve(vocab_size);
    loaded_dfs.reserve(vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
      uint32_t len = 0;
      is.read(reinterpret_cast<char *>(&len), sizeof(len));
      std::string token(len, '\0');
      is.read(token.data(), len);
      uint32_t df = 0;
      is.read(reinterpret_cast<char *>(&df), sizeof(df));
      loaded_tokens.push_back(std::move(token));
      loaded_dfs.push_back(df);
    }

    // Inject exact mapping
    vocab_.set_state(std::move(loaded_tokens), std::move(loaded_dfs));

    // Load IDF weights
    uint64_t idf_size = 0;
    is.read(reinterpret_cast<char *>(&idf_size), sizeof(idf_size));
    std::vector<float> loaded_idf(idf_size);
    is.read(reinterpret_cast<char *>(loaded_idf.data()), sizeof(float) * idf_size);

    // Inject exact weights
    tfidf_.set_idf_weights(std::move(loaded_idf));

    // Load tokenizer options
    uint32_t ngram_min = 0;
    uint32_t ngram_max = 0;
    is.read(reinterpret_cast<char *>(&ngram_min), sizeof(ngram_min));
    is.read(reinterpret_cast<char *>(&ngram_max), sizeof(ngram_max));
    tokenizer_options_.ngram_min = static_cast<size_t>(ngram_min);
    tokenizer_options_.ngram_max = static_cast<size_t>(ngram_max);
    uint8_t lowercase = 0;
    uint8_t split_camel_case = 0;
    uint8_t preserve_operators = 0;
    is.read(reinterpret_cast<char *>(&lowercase), sizeof(lowercase));
    is.read(reinterpret_cast<char *>(&split_camel_case), sizeof(split_camel_case));
    is.read(reinterpret_cast<char *>(&preserve_operators), sizeof(preserve_operators));
    tokenizer_options_.lowercase = (lowercase != 0);
    tokenizer_options_.split_camel_case = (split_camel_case != 0);
    tokenizer_options_.preserve_operators = (preserve_operators != 0);
    tokenizer_ = std::make_unique<Tokenizer>(tokenizer_options_);

    // Load model weights
    lr_.load(is);

    uint8_t mt = 0;
    is.read(reinterpret_cast<char *>(&mt), sizeof(mt));
    model_type_ = static_cast<ModelType>(mt);
    if (model_type_ == ModelType::ResidualMLP) {
      mlp_.load(is);
    }

    // Load classification thresholds
    is.read(reinterpret_cast<char *>(&thresholds_), sizeof(ClassificationThresholds));
  }

  bool EmbeddedClassifier::save_to_file(std::string_view filepath) const
  {
    std::ofstream ofs(std::string(filepath), std::ios::binary);
    if (!ofs.is_open()) return false;
    save(ofs);
    return true;
  }

  bool EmbeddedClassifier::load_from_file(std::string_view filepath)
  {
    std::ifstream ifs(std::string(filepath), std::ios::binary);
    if (!ifs.is_open()) return false;
    load(ifs);
    return true;
  }

} // namespace classifier
