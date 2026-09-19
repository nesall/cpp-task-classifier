Config params used to train `baseline_v2` model.
No MLP is used inside `EmbeddedClassifier::classify`



```

  TokenizerOptions tok_opts;
  tok_opts.ngram_min = 1;
  tok_opts.ngram_max = 1;
  tok_opts.split_camel_case = true;
  tok_opts.preserve_operators = true;

  ClassificationThresholds thresholds;
  thresholds.tier3_margin = 0.12f;
  thresholds.tier2_margin = 0.22f;
  thresholds.min_t3_prob = 0.30f;
  thresholds.min_t2_prob = 0.35f;

  // Toggle between ModelType::Linear and ModelType::ResidualMLP here:
  EmbeddedClassifier classifier(
    ModelType::ResidualMLP,
    tok_opts,
    thresholds
  );

  TrainConfig linear_cfg;
  linear_cfg.learning_rate = 0.5f;
  linear_cfg.l2_reg = 1e-5f;
  linear_cfg.epochs = 150;
  linear_cfg.batch_size = 32;
  linear_cfg.verbose = false;

  std::array<size_t, 3> counts{ 0, 0, 0 };
  for (auto t : train_ds.labels) counts[static_cast<size_t>(t)]++;
  for (size_t k = 0; k < 3; ++k) {
    linear_cfg.class_weights[k] = std::sqrt(static_cast<float>(train_ds.size()) / (3.0f * static_cast<float>(counts[k])));
    std::cout << "class_weight for " << to_string(static_cast<Tier>(k)) << ": " << linear_cfg.class_weights[k] << "\n";
  }

  ResidualConfig res_cfg;
  res_cfg.hidden_dim = 8;       // Smaller bottleneck (8 units = ~120k parameters)
  res_cfg.learning_rate = 0.2f;
  res_cfg.l2_reg = 1e-3f;   // Stronger L2 regularization to prevent memorization (was 1e-5)
  res_cfg.epochs = 45;      // Stop before overfitting (was 100)
  res_cfg.batch_size = 32;

  classifier.fit(
    train_ds.texts,
    train_ds.labels,
    15000,
    2,
    linear_cfg,
    res_cfg
  );

```

