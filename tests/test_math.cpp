#include "classifier/metrics.h"
#include <cmath>

using namespace classifier;

static bool approx_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) < eps;
}

bool test_metrics_perfect_classification() {
  std::vector<Tier> truth = {
      Tier::Tier1Simple, Tier::Tier2Medium, Tier::Tier3Complex,
      Tier::Tier1Simple, Tier::Tier2Medium, Tier::Tier3Complex
  };
  std::vector<Tier> preds = truth;

  auto report = Metrics::evaluate(truth, preds);

  if (!approx_equal(report.accuracy, 1.0f)) return false;
  if (!approx_equal(report.macro_precision, 1.0f)) return false;
  if (!approx_equal(report.macro_recall, 1.0f)) return false;
  if (!approx_equal(report.macro_f1, 1.0f)) return false;

  for (size_t i = 0; i < 3; ++i) {
    if (report.confusion_matrix[i][i] != 2) return false;
  }

  return true;
}

bool test_metrics_known_confusion_matrix() {
  // 6 samples:
  // True: [T1, T1, T2, T2, T3, T3]
  // Pred: [T1, T2, T2, T3, T3, T3]
  std::vector<Tier> truth = {
      Tier::Tier1Simple, Tier::Tier1Simple,
      Tier::Tier2Medium, Tier::Tier2Medium,
      Tier::Tier3Complex, Tier::Tier3Complex
  };
  std::vector<Tier> preds = {
      Tier::Tier1Simple, Tier::Tier2Medium,
      Tier::Tier2Medium, Tier::Tier3Complex,
      Tier::Tier3Complex, Tier::Tier3Complex
  };

  auto report = Metrics::evaluate(truth, preds);

  // Accuracy: 4/6 = 0.6667
  if (!approx_equal(report.accuracy, 4.0f / 6.0f)) return false;

  // T1: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1 = 2*(1*0.5)/(1.5) = 2/3 = 0.6667
  if (!approx_equal(report.per_class[0].precision, 1.0f)) return false;
  if (!approx_equal(report.per_class[0].recall, 0.5f)) return false;
  if (!approx_equal(report.per_class[0].f1_score, 2.0f / 3.0f)) return false;

  // T2: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1 = 0.5
  if (!approx_equal(report.per_class[1].precision, 0.5f)) return false;
  if (!approx_equal(report.per_class[1].recall, 0.5f)) return false;
  if (!approx_equal(report.per_class[1].f1_score, 0.5f)) return false;

  // T3: TP=2, FP=1, FN=0 -> P=2/3 = 0.6667, R=1.0, F1 = 2*(2/3 * 1)/(5/3) = 4/5 = 0.8
  if (!approx_equal(report.per_class[2].precision, 2.0f / 3.0f)) return false;
  if (!approx_equal(report.per_class[2].recall, 1.0f)) return false;
  if (!approx_equal(report.per_class[2].f1_score, 0.8f)) return false;

  // Macro F1: (2/3 + 1/2 + 4/5) / 3 = (0.6667 + 0.5 + 0.8) / 3 = 1.9667 / 3 = 0.6556
  float expected_macro_f1 = ((2.0f / 3.0f) + 0.5f + 0.8f) / 3.0f;
  if (!approx_equal(report.macro_f1, expected_macro_f1)) return false;

  return true;
}
