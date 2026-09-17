#include "classifier/tokenizer.h"
#include <algorithm>
#include <vector>
#include <string>

using namespace classifier;

bool test_tokenizer_preserves_scoped_identifiers() {
  Tokenizer tokenizer;
  std::string text = "std::vector<int> and Foo::Bar::process()";
  std::vector<std::string> tokens = tokenizer.tokenize(text);

  // Must preserve scoped tokens
  auto has_token = [&](const std::string &target) {
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
    };

  if (!has_token("std::vector")) return false;
  if (!has_token("foo::bar::process")) return false;
  if (!has_token("vector")) return false;
  if (!has_token("process")) return false;

  return true;
}

bool test_tokenizer_handles_camel_case() {
  Tokenizer tokenizer;
  std::string text = "ParseHttpRequest and m_bufferSize";
  std::vector<std::string> tokens = tokenizer.tokenize(text);

  auto has_token = [&](const std::string &target) {
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
    };

  if (!has_token("parsehttprequest")) return false;
  if (!has_token("parse")) return false;
  if (!has_token("http")) return false;
  if (!has_token("request")) return false;
  if (!has_token("buffer")) return false;
  if (!has_token("size")) return false;

  return true;
}

bool test_tokenizer_preserves_code_operators() {
  Tokenizer tokenizer;
  std::string text = "ptr->value == *ref && val != nullptr";
  std::vector<std::string> tokens = tokenizer.tokenize(text);

  auto has_token = [&](const std::string &target) {
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
    };

  if (!has_token("->")) return false;
  if (!has_token("==")) return false;
  if (!has_token("*")) return false;
  if (!has_token("&&")) return false;
  if (!has_token("!=")) return false;

  return true;
}

bool test_tokenizer_extracts_bigrams() {
  TokenizerOptions opts;
  opts.ngram_min = 1;
  opts.ngram_max = 2;
  opts.split_camel_case = false;
  Tokenizer tokenizer(opts);

  std::string text = "binary search tree";
  auto tokens = tokenizer.tokenize(text);

  auto has_token = [&](const std::string &target) {
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
    };

  if (!has_token("binary")) return false;
  if (!has_token("search")) return false;
  if (!has_token("tree")) return false;
  if (!has_token("binary_search")) return false; // Bigram 1
  if (!has_token("search_tree")) return false;   // Bigram 2

  return true;
}

bool test_tokenizer_ngrams_with_operators() {
  TokenizerOptions opts;
  opts.ngram_min = 2;
  opts.ngram_max = 3;
  opts.split_camel_case = false;
  Tokenizer tokenizer(opts);

  std::string text = "vector < int >";
  auto tokens = tokenizer.tokenize(text);

  auto has_token = [&](const std::string &target) {
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
    };

  if (!has_token("vector_<")) return false;
  if (!has_token("<_int")) return false;
  if (!has_token("int_>")) return false;
  if (!has_token("vector_<_int")) return false;

  return true;
}

bool test_tokenizer_filters_stop_words() {
  TokenizerOptions opts;
  opts.ngram_min = 1;
  opts.ngram_max = 1;
  Tokenizer tokenizer(opts);

  std::string text = "What does std::move do in C++?";
  auto tokens = tokenizer.tokenize(text);

  auto has_token = [&](std::string_view target) {
    return std::find(tokens.begin(), tokens.end(), target) != tokens.end();
    };

  // Stop words dropped
  if (has_token("what") || has_token("does") || has_token("do") || has_token("in")) {
    return false;
  }

  // Code & domain terms preserved
  if (!has_token("std::move") || !has_token("move") || !has_token("c++")) {
    return false;
  }

  return true;
}