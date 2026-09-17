#include <iostream>
#include <vector>
#include <functional>
#include <string>

namespace test {

  struct TestCase {
    std::string name;
    std::function<bool()> func;
  };

  std::vector<TestCase> &registry() {
    static std::vector<TestCase> cases;
    return cases;
  }

  void register_test(std::string name, std::function<bool()> func) {
    registry().push_back({ std::move(name), std::move(func) });
  }

  int run_all() {
    int passed = 0;
    int failed = 0;

    for (const auto &tc : registry()) {
      std::cout << "[ RUN      ] " << tc.name << "\n";
      bool ok = false;
      try {
        ok = tc.func();
      } catch (const std::exception &e) {
        std::cout << "  Exception thrown: " << e.what() << "\n";
        ok = false;
      } catch (...) {
        std::cout << "  Unknown non-std exception thrown\n";
        ok = false;
      }

      if (ok) {
        std::cout << "[   PASSED ] " << tc.name << "\n";
        passed++;
      } else {
        std::cout << "[   FAILED ] " << tc.name << "\n";
        failed++;
      }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "Total: " << (passed + failed) << " | Passed: " << passed << " | Failed: " << failed << "\n";
    std::cout << "----------------------------------------\n";

    return (failed == 0) ? 0 : 1;
  }

} // namespace test

#define REGISTER_TEST(name) \
    bool test_##name(); \
    static const bool reg_##name = []() { \
        test::register_test(#name, test_##name); \
        return true; \
    }(); \
    bool test_##name()

// Forward-declared tests
REGISTER_TEST(tokenizer_preserves_scoped_identifiers);
REGISTER_TEST(tokenizer_handles_camel_case);
REGISTER_TEST(tokenizer_preserves_code_operators);
REGISTER_TEST(tokenizer_extracts_bigrams);
REGISTER_TEST(tokenizer_ngrams_with_operators);
REGISTER_TEST(tokenizer_filters_stop_words);
REGISTER_TEST(metrics_perfect_classification);
REGISTER_TEST(metrics_known_confusion_matrix);
REGISTER_TEST(vocabulary_pruning_and_order);
REGISTER_TEST(tfidf_exact_weights_and_normalization);
REGISTER_TEST(tfidf_oov_tokens);
REGISTER_TEST(softmax_numerical_stability);
REGISTER_TEST(logistic_gradient_check);
REGISTER_TEST(logistic_overfitting_toy_data);
REGISTER_TEST(logistic_serialization_roundtrip);
REGISTER_TEST(embedded_classifier_pipeline);
REGISTER_TEST(neural_gradient_check);
REGISTER_TEST(neural_overfits_toy_dataset);

int main() {
  return test::run_all();
}
