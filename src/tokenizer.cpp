#include "classifier/tokenizer.h"
#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace classifier {

  namespace {

    inline bool is_ident_start(char c) noexcept {
      return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    }

    inline bool is_ident_char(char c) noexcept {
      return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    inline std::string to_lower(std::string_view s) {
      std::string res;
      res.reserve(s.size());
      for (char c : s) {
        res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
      }
      return res;
    }

    inline bool is_stop_word(std::string_view s) noexcept {
      static const std::unordered_set<std::string_view> STOP_WORDS = {
          "a", "about", "above", "after", "again", "all", "am", "an", "and",
          "any", "are", "aren", "as", "at", "be", "because", "been", "before",
          "being", "below", "between", "both", "but", "by", "can", "could",
          "did", "didn", "do", "does", "doesn", "doing", "don", "down",
          "during", "each", "few", "for", "from", "further", "had", "hadn",
          "has", "hasn", "have", "haven", "having", "he", "her", "here",
          "hers", "herself", "him", "himself", "his", "how", "i", "if",
          "in", "into", "is", "isn", "it", "its", "itself", "just", "ll",
          "m", "ma", "me", "might", "more", "most", "must", "my", "myself",
          "need", "no", "nor", "not", "now", "o", "of", "off", "on", "once",
          "only", "or", "other", "our", "ours", "ourselves", "out", "over",
          "own", "re", "s", "same", "she", "should", "so", "some", "such",
          "t", "than", "that", "the", "their", "theirs", "them", "themselves",
          "then", "there", "these", "they", "this", "those", "through", "to",
          "too", "under", "until", "up", "ve", "very", "was", "wasn", "we",
          "were", "weren", "what", "when", "where", "which", "while", "who",
          "whom", "why", "will", "with", "won", "would", "y", "you", "your",
          "yours", "yourself", "yourselves", "want",
          "implement", "create", "write", "optimize", "modify"
      };
      return STOP_WORDS.contains(s);
    }

  } // namespace

  void Tokenizer::split_and_append_subwords(std::string_view token, std::vector<std::string> &out) const {
    if (!options_.split_camel_case || token.empty()) return;

    size_t start = 0;
    for (size_t i = 1; i < token.size(); ++i) {
      bool is_snake = (token[i] == '_');
      bool is_camel = std::isupper(static_cast<unsigned char>(token[i])) &&
        !std::isupper(static_cast<unsigned char>(token[i - 1]));

      if (is_snake || is_camel) {
        if (i > start) {
          std::string_view sub = token.substr(start, i - start);
          if (sub != "_") {
            std::string s = options_.lowercase ? to_lower(sub) : std::string(sub);
            if (!is_stop_word(s)) {
              out.push_back(std::move(s));
            }
          }
        }
        start = is_snake ? i + 1 : i;
      }
    }

    if (start < token.size()) {
      std::string_view sub = token.substr(start);
      if (sub != "_") {
        std::string s = options_.lowercase ? to_lower(sub) : std::string(sub);
        if (!is_stop_word(s)) {
          out.push_back(std::move(s));
        }
      }
    }
  }

  std::vector<std::string> Tokenizer::tokenize(std::string_view text) const {
    std::vector<std::string> sequence;
    std::vector<std::string> subwords;
    sequence.reserve(64);
    subwords.reserve(32);

    const size_t n = text.size();
    size_t i = 0;

    while (i < n) {
      if (std::isspace(static_cast<unsigned char>(text[i]))) {
        ++i;
        continue;
      }

      if (is_ident_start(text[i])) {
        size_t start = i;
        while (i < n) {
          if (is_ident_char(text[i])) {
            ++i;
          } else if (i + 1 < n && text[i] == ':' && text[i + 1] == ':') {
            i += 2;
          } else {
            break;
          }
        }
        std::string_view raw = text.substr(start, i - start);
        std::string tok = options_.lowercase ? to_lower(raw) : std::string(raw);

        if (options_.split_camel_case) {
          size_t sub_start = 0;
          while (sub_start < raw.size()) {
            size_t scope_pos = raw.find("::", sub_start);
            std::string_view part = (scope_pos == std::string_view::npos)
              ? raw.substr(sub_start)
              : raw.substr(sub_start, scope_pos - sub_start);

            split_and_append_subwords(part, subwords);

            if (scope_pos == std::string_view::npos) break;
            sub_start = scope_pos + 2;
          }
        }

        if (!is_stop_word(tok)) {
          sequence.push_back(std::move(tok));
        }
        continue;
      }

      if (options_.preserve_operators && i + 1 < n) {
        std::string_view two_char = text.substr(i, 2);
        if (two_char == "::" || two_char == "->" || two_char == "==" ||
          two_char == "!=" || two_char == "<=" || two_char == ">=" ||
          two_char == "&&" || two_char == "||" || two_char == "++" || two_char == "--") {
          sequence.emplace_back(two_char);
          i += 2;
          continue;
        }
      }

      char c = text[i];
      if (options_.preserve_operators && (c == '*' || c == '&' || c == '~' || c == '<' || c == '>')) {
        sequence.emplace_back(1, c);
        ++i;
        continue;
      }

      if (std::isdigit(static_cast<unsigned char>(c))) {
        size_t start = i;
        while (i < n && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '.')) {
          ++i;
        }
        sequence.emplace_back(text.substr(start, i - start));
        continue;
      }
      ++i;
    }

    std::vector<std::string> final_tokens;
    const size_t seq_len = sequence.size();
    final_tokens.reserve(seq_len + (options_.ngram_max > 1 ? seq_len : 0) + subwords.size());

    // 1. Emit filtered unigrams
    if (options_.ngram_min <= 1) {
      for (const auto &tok : sequence) {
        final_tokens.push_back(tok);
      }
    }

    // 2. Emit bigrams with a reusable scratch string to avoid continuous allocations
    if (options_.ngram_max >= 2 && seq_len >= 2) {
      std::string scratch;
      scratch.reserve(64);

      for (size_t idx = 0; idx < seq_len - 1; ++idx) {
        scratch.clear();
        scratch.append(sequence[idx]);
        scratch.push_back('_');
        scratch.append(sequence[idx + 1]);
        final_tokens.push_back(scratch);
      }
    }

    // 3. Emit subwords
    for (auto &sw : subwords) {
      final_tokens.push_back(std::move(sw));
    }

    return final_tokens;
  }

} // namespace classifier