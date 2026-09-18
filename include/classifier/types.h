#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace classifier {

  enum class Tier : uint8_t {
    Tier1Simple = 0,
    Tier2Medium = 1,
    Tier3Complex = 2,
    Unknown = 255
  };

  [[nodiscard]] constexpr std::string_view to_string(Tier tier) noexcept {
    switch (tier) {
    case Tier::Tier1Simple:   return "TIER_1_SIMPLE";
    case Tier::Tier2Medium: return "TIER_2_MEDIUM";
    case Tier::Tier3Complex:  return "TIER_3_COMPLEX";
    default:                  return "UNKNOWN";
    }
  }

  [[nodiscard]] inline Tier tier_from_string(std::string_view str) noexcept {
    if (str == "TIER_1_SIMPLE" || str == "0") return Tier::Tier1Simple;
    if (str == "TIER_2_MEDIUM" || str == "1") return Tier::Tier2Medium;
    if (str == "TIER_3_COMPLEX" || str == "2") return Tier::Tier3Complex;
    return Tier::Unknown;
  }

  struct SparseFeature {
    uint32_t index{ 0 };
    float value{ 0.0f };
  };

  using SparseVector = std::vector<SparseFeature>;

  struct ClassificationResult {
    Tier predicted_tier{ Tier::Unknown };
    std::array<float, 3> probabilities{ 0.0f, 0.0f, 0.0f };
    float confidence{ 0.0f };
  };

  struct LabeledExample {
    std::string id;
    std::string text;
    Tier label{ Tier::Unknown };
  };

} // namespace classifier