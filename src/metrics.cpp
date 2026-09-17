#include "classifier/metrics.h"
#include <cassert>
#include <iomanip>
#include <sstream>

namespace classifier {

  EvaluationReport Metrics::evaluate(
    std::span<const Tier> ground_truth,
    std::span<const Tier> predictions
  )
  {
    assert(ground_truth.size() == predictions.size());

    EvaluationReport report{};
    report.total_samples = ground_truth.size();
    if (report.total_samples == 0) {
      return report;
    }

    size_t correct_predictions = 0;

    for (size_t i = 0; i < ground_truth.size(); ++i) {
      const auto true_idx = static_cast<size_t>(ground_truth[i]);
      const auto pred_idx = static_cast<size_t>(predictions[i]);

      if (true_idx < 3 && pred_idx < 3) {
        report.confusion_matrix[true_idx][pred_idx]++;
        if (true_idx == pred_idx) {
          correct_predictions++;
        }
      }
    }

    report.accuracy = static_cast<float>(correct_predictions) / static_cast<float>(report.total_samples);

    float sum_p = 0.0f;
    float sum_r = 0.0f;
    float sum_f1 = 0.0f;

    for (size_t k = 0; k < 3; ++k) {
      size_t tp = report.confusion_matrix[k][k];
      size_t fp = 0;
      size_t fn = 0;
      size_t support = 0;

      for (size_t row = 0; row < 3; ++row) {
        if (row != k) fp += report.confusion_matrix[row][k];
        support += report.confusion_matrix[k][row];
      }

      for (size_t col = 0; col < 3; ++col) {
        if (col != k) fn += report.confusion_matrix[k][col];
      }

      ClassMetrics &cm = report.per_class[k];
      cm.support = support;
      cm.precision = (tp + fp > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
      cm.recall = (tp + fn > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
      cm.f1_score = (cm.precision + cm.recall > 0.0f)
        ? (2.0f * cm.precision * cm.recall) / (cm.precision + cm.recall)
        : 0.0f;

      sum_p += cm.precision;
      sum_r += cm.recall;
      sum_f1 += cm.f1_score;
    }

    report.macro_precision = sum_p / 3.0f;
    report.macro_recall = sum_r / 3.0f;
    report.macro_f1 = sum_f1 / 3.0f;

    return report;
  }

  std::string EvaluationReport::to_string() const
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    oss << "Confusion Matrix (rows: True, cols: Pred):\n";
    oss << "          T1_SIMP  T2_REFA  T3_COMP\n";
    for (size_t r = 0; r < 3; ++r) {
      oss << "  " << std::left << std::setw(7) << classifier::to_string(static_cast<Tier>(r)).substr(0, 7) << " ";
      for (size_t c = 0; c < 3; ++c) {
        oss << std::right << std::setw(8) << confusion_matrix[r][c] << " ";
      }
      oss << "\n";
    }

    oss << "\nPer-Class Metrics:\n";
    oss << "  Class             Precision  Recall   F1-Score  Support\n";
    for (size_t k = 0; k < 3; ++k) {
      oss << "  " << std::left << std::setw(17) << classifier::to_string(static_cast<Tier>(k))
        << std::right << std::setw(9) << per_class[k].precision << "  "
        << std::setw(6) << per_class[k].recall << "  "
        << std::setw(8) << per_class[k].f1_score << "  "
        << std::setw(7) << per_class[k].support << "\n";
    }

    oss << "\nSummary:\n";
    oss << "  Accuracy:        " << accuracy << "\n";
    oss << "  Macro Precision: " << macro_precision << "\n";
    oss << "  Macro Recall:    " << macro_recall << "\n";
    oss << "  Macro F1:        " << macro_f1 << "\n";

    return oss.str();
  }

} // namespace classifier
