#include "classifier/dataset.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace classifier {

  using json = nlohmann::json;

  void Dataset::clear()
  {
    ids.clear();
    texts.clear();
    labels.clear();
  }

  Dataset DatasetLoader::load_jsonl(std::string_view filepath)
  {
    std::ifstream file{ std::string(filepath) };
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open JSONL file: " + std::string(filepath));
    }

    Dataset ds;
    std::string line;
    size_t line_num = 0;

    while (std::getline(file, line)) {
      line_num++;
      if (line.empty()) continue;

      try {
        auto j = json::parse(line);
        std::string id = j.value("id", "item_" + std::to_string(line_num));
        std::string text = j.at("text").get<std::string>();
        std::string label_str = j.at("label").get<std::string>();

        Tier tier = tier_from_string(label_str);
        if (tier == Tier::Unknown) {
          continue; // Skip unrecognized labels
        }

        ds.ids.push_back(std::move(id));
        ds.texts.push_back(std::move(text));
        ds.labels.push_back(tier);
      } catch (const std::exception &e) {
        // Malformed JSON lines can be skipped or logged
        continue;
      }
    }

    return ds;
  }

  void DatasetLoader::save_jsonl(std::string_view filepath, const Dataset &dataset)
  {
    std::ofstream file{ std::string(filepath) };
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open output file: " + std::string(filepath));
    }

    for (size_t i = 0; i < dataset.size(); ++i) {
      json j;
      j["id"] = dataset.ids[i];
      j["text"] = dataset.texts[i];
      j["label"] = std::string(to_string(dataset.labels[i]));
      file << j.dump() << "\n";
    }
  }

} // namespace classifier
