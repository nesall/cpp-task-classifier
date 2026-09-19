#include "classifier/model.h"
#include "classifier/types.h"
#include <string>
#include <cmath>
#include <sstream>

using namespace classifier;

bool test_embedded_classifier_pipeline() {
  std::vector<std::string> texts = {
      "What does std::move do?",
      "How to check empty string?",
      "Refactor Foo::bar across all files",
      "Rename database column and update queries",
      "Diagnose deadlock in multithreaded lock-free queue",
      "Design distributed consensus with Byzantine fault tolerance"
  };

  std::vector<Tier> labels = {
      Tier::Tier1Simple,
      Tier::Tier1Simple,
      Tier::Tier2Medium,
      Tier::Tier2Medium,
      Tier::Tier3Complex,
      Tier::Tier3Complex
  };

  TrainConfig config;
  config.learning_rate = 0.5f;
  config.l2_reg = 0.0f;
  config.epochs = 100;
  config.batch_size = 6;
  config.random_seed = 42;

  EmbeddedClassifier clf;
  clf.fit(texts, labels, 100, 1, config);

  // Verify self-consistency
  for (size_t i = 0; i < texts.size(); ++i) {
    auto res = clf.classify(texts[i]);
    if (res.predicted_tier != labels[i]) {
      return false;
    }
  }

  // Verify serialization roundtrip
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  clf.save(ss);

  EmbeddedClassifier loaded_clf;
  ss.seekg(0);
  loaded_clf.load(ss);

  for (size_t i = 0; i < texts.size(); ++i) {
    auto res_orig = clf.classify(texts[i]);
    auto res_loaded = loaded_clf.classify(texts[i]);

    if (res_orig.predicted_tier != res_loaded.predicted_tier) return false;
    for (size_t k = 0; k < 3; ++k) {
      if (std::fabs(res_orig.probabilities[k] - res_loaded.probabilities[k]) > 1e-4f) {
        return false;
      }
    }
  }

  return true;
}
