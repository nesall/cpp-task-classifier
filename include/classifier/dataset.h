#pragma once

#include "classifier/types.h"
#include <string>
#include <string_view>
#include <vector>

namespace classifier {

  struct Dataset {
    std::vector<std::string> ids;
    std::vector<std::string> texts;
    std::vector<Tier> labels;

    [[nodiscard]] size_t size() const noexcept { return texts.size(); }
    [[nodiscard]] bool empty() const noexcept { return texts.empty(); }
    void clear();
  };

  class DatasetLoader {
  public:
    [[nodiscard]] static Dataset load_jsonl(std::string_view filepath);
    static void save_jsonl(std::string_view filepath, const Dataset &dataset);
  };

} // namespace classifier
