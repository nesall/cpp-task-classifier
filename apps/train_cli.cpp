#include "classifier/dataset.h"
#include "classifier/metrics.h"
#include "classifier/model.h"
#include "classifier/neural_classifier.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;
fs::path RootDir = PROJECT_ROOT_DIR;

using namespace classifier;

void evaluate_dataset(const EmbeddedClassifier &classifier, const Dataset &ds, const std::string &split_name) {
  if (ds.empty()) return;

  std::vector<Tier> predictions;
  predictions.reserve(ds.size());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (const auto &text : ds.texts) {
    predictions.push_back(classifier.classify(text).predicted_tier);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double avg_us = total_us / static_cast<double>(ds.size());

  auto report = Metrics::evaluate(ds.labels, predictions);

  std::cout << "========================================================\n";
  std::cout << " Split: " << split_name << " (Samples: " << ds.size() << ")\n";
  std::cout << " Inference: " << avg_us << " us/query\n";
  std::cout << "========================================================\n";
  std::cout << report.to_string() << "\n";
}

void inspect_top_features(const EmbeddedClassifier &classifier, size_t top_k = 15) {
  const auto &vocab = classifier.vocabulary();
  const auto &model = classifier.linear_model();
  const size_t n_feats = vocab.size();

  std::cout << "\n================ TOP WEIGHT INSPECTION ================\n";
  for (size_t k = 0; k < 3; ++k) {
    std::vector<std::pair<float, std::string_view>> feat_weights;
    feat_weights.reserve(n_feats);

    for (size_t j = 0; j < n_feats; ++j) {
      feat_weights.emplace_back(model.get_weight(k, j), vocab.get_token(static_cast<uint32_t>(j)));
    }

    std::sort(feat_weights.begin(), feat_weights.end(), [](const auto &a, const auto &b) {
      return a.first > b.first;
      });

    std::cout << "\nTop Features for " << to_string(static_cast<Tier>(k)) << ":\n";
    for (size_t i = 0; i < std::min(top_k, feat_weights.size()); ++i) {
      std::cout << "  " << std::setw(25) << std::left << feat_weights[i].second
        << " : " << std::fixed << std::setprecision(4) << feat_weights[i].first << "\n";
    }
  }
  std::cout << "=======================================================\n\n";
}

int main(int argc, char *argv[]) {
  fs::path train_path = RootDir / "data/processed/train.jsonl";
  fs::path val_path = RootDir / "data/processed/val.jsonl";
  fs::path test_path = RootDir / "data/processed/test.jsonl";
  fs::path bench_path = RootDir / "data/benchmarks/benchmark_v0.1.jsonl";
  fs::path out_model = RootDir / "models/baseline_v1.bin";

  if (argc >= 2) train_path = argv[1];
  if (argc >= 3) val_path = argv[2];
  if (argc >= 4) test_path = argv[3];

  std::cout << "Loading datasets...\n";
  Dataset train_ds, val_ds, test_ds, bench_ds;
  try {
    train_ds = DatasetLoader::load_jsonl(train_path.string());
    std::cout << "Train dataset loaded: " << train_ds.size() << " samples.\n";
    val_ds = DatasetLoader::load_jsonl(val_path.string());
    std::cout << "Validation dataset loaded: " << val_ds.size() << " samples.\n";
    test_ds = DatasetLoader::load_jsonl(test_path.string());
    std::cout << "Test dataset loaded: " << test_ds.size() << " samples.\n";
    bench_ds = DatasetLoader::load_jsonl(bench_path.string());
    std::cout << "Benchmark v0.1 loaded: " << bench_ds.size() << " samples.\n";
  } catch (const std::exception &e) {
    std::cerr << "Dataset loading error: " << e.what() << "\n";
    return 1;
  }

  TokenizerOptions tok_opts;
  tok_opts.ngram_min = 1;
  tok_opts.ngram_max = 2;
  tok_opts.split_camel_case = true;
  tok_opts.preserve_operators = true;

  ClassificationThresholds thresholds;
  thresholds.tier3_margin = 0.08f;
  thresholds.tier2_margin = 0.05f;
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

  inspect_top_features(classifier, 15);

  std::cout << "\nSaving model to " << out_model << "...\n";
  classifier.save_to_file(out_model.string());

  std::cout << "\nEvaluating model performance:\n";
  evaluate_dataset(classifier, train_ds, "TRAIN");
  evaluate_dataset(classifier, val_ds, "VALIDATION");
  evaluate_dataset(classifier, test_ds, "TEST");
  evaluate_dataset(classifier, bench_ds, "BENCHMARK v0.1 (Regression Suite)");

  std::cout << "\n--- ERROR ANALYSIS (BENCHMARK) ---\n";
  for (size_t i = 0; i < bench_ds.size(); ++i) {
    auto res = classifier.classify(bench_ds.texts[i]);
    if (res.predicted_tier != bench_ds.labels[i]) {
      std::cout << "MISCLASSIFIED:\n"
        << "  Text: " << bench_ds.texts[i] << "\n"
        << "  True: " << to_string(bench_ds.labels[i]) << "\n"
        << "  Pred: " << to_string(res.predicted_tier) << "\n"
        << "  Conf: T1=" << res.probabilities[0]
        << " | T2=" << res.probabilities[1]
        << " | T3=" << res.probabilities[2] << "\n\n";
    }
  }
  return 0;
}
