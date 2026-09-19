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


void test_context_inheritance(const classifier::EmbeddedClassifier &classifier) {
  using classifier::Tier;
  std::cout << "Running test_context_inheritance...\n";
  bool ok = true;
  std::string query;
  // Scenario 1: Short follow-up with Tier 3 Context
  // Token length is 2. context_weight = max(0, 1 - (2/15)) = ~0.86
  // Logit boost = 2.5 * 0.86 = +2.15 to Tier 3. 
  // This aggressively forces the ambiguous query into Tier 3.
  auto result_t3 = classifier.classify(query = "How so?", Tier::Tier3Complex);
  if (result_t3.predicted_tier != Tier::Tier3Complex) {
    std::cout << "Short query \"" << query << "\" failed to inherit Tier 3 context\n";
    ok = false;
  }

  // Scenario 2: Short follow-up with Tier 2 Context
  // Token length is 2. Same math, but forces into Tier 2.

  auto result_t2 = classifier.classify(query = "Explain that.", Tier::Tier2Medium);
  if (result_t2.predicted_tier != Tier::Tier2Medium) {
    std::cout << "Short query \"" << query << "\" failed to inherit Tier 2 context\n";
    ok = false;
  }

  // Scenario 3: Context Override via Long Query
  // Token length is ~17. context_weight = max(0, 1 - (17/15)) = 0.0
  // Logit boost = +0.0. The model must rely purely on lexical TF-IDF.
  // "std::vector" and "How do I" will trigger the Tier 1 syntax guard.
  std::string long_t1_query = "How do I iterate over a std::vector<int> using a range-based for loop in modern C++?";

  // We pass Tier3Complex as the previous session context, simulating a subject change mid-chat.
  auto result_override = classifier.classify(long_t1_query, Tier::Tier3Complex);
  if (result_override.predicted_tier != Tier::Tier1Simple) {
    std::cout << "Long explicit query \"" << long_t1_query << "\" failed to override session context\n";
    ok = false;
  }

  if (ok)
    std::cout << "  -> test_context_inheritance PASSED\n";
  else 
    std::cout << "  -> test_context_inheritance FAILED\n";
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

#if 0
  std::cout << "\n--- T1/T2 CONFUSIONS (TRAIN) ---\n";
  std::vector<std::pair<double, size_t>> t1_pred_t2; // (T2 prob, index) - true T1, pred T2
  std::vector<std::pair<double, size_t>> t2_pred_t1; // (T1 prob, index) - true T2, pred T1
  for (size_t i = 0; i < train_ds.size(); ++i) {
    auto res = classifier.classify(train_ds.texts[i]);
    Tier true_t = train_ds.labels[i];
    Tier pred_t = res.predicted_tier;
    if (true_t == Tier::Tier1Simple && pred_t == Tier::Tier2Medium)
      t1_pred_t2.emplace_back(res.probabilities[1], i);
    else if (true_t == Tier::Tier2Medium && pred_t == Tier::Tier1Simple)
      t2_pred_t1.emplace_back(res.probabilities[0], i);
  }
  std::cout << "\nT1->T2 confusions: " << t1_pred_t2.size() << " | T2->T1 confusions: " << t2_pred_t1.size() << "\n";
  // Sort by confidence descending (most confidently wrong first) and print top 15 of each
  auto print_top = [&](std::vector<std::pair<double, size_t>> &v, const char *label) {
    std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first > b.first; });
    std::cout << "\n--- Top " << label << " (most confident misses) ---\n";
    for (size_t i = 0; i < std::min<size_t>(15, v.size()); ++i) {
      std::cout << "  [conf=" << v[i].first << "] " << train_ds.texts[v[i].second] << "\n";
    }
    };
  print_top(t1_pred_t2, "TRUE T1, PRED T2");
  print_top(t2_pred_t1, "TRUE T2, PRED T1");
#endif

  test_context_inheritance(classifier);

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
