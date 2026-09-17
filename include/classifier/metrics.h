#pragma once

#include "classifier/types.h"
#include <array>
#include <cstddef>
#include <span>
#include <string>

namespace classifier {

  struct ClassMetrics {
    float precision{ 0.0f };
    float recall{ 0.0f };
    float f1_score{ 0.0f };
    size_t support{ 0 };
  };

  struct EvaluationReport {
    std::array<std::array<size_t, 3>, 3> confusion_matrix{};
    std::array<ClassMetrics, 3> per_class{};
    float accuracy{ 0.0f };
    float macro_precision{ 0.0f };
    float macro_recall{ 0.0f };
    float macro_f1{ 0.0f };
    size_t total_samples{ 0 };

    [[nodiscard]] std::string to_string() const;
  };

  class Metrics {
  public:
    [[nodiscard]] static EvaluationReport evaluate(
      std::span<const Tier> ground_truth,
      std::span<const Tier> predictions
    );
  };

} // namespace classifier