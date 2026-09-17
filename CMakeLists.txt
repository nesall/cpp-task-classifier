cmake_minimum_required(VERSION 3.20)
project(classifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if (MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  add_compile_options(/MP /GS /W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

include_directories(
  include
  include/3rdparty
)

# Core library
add_library(classifier_core STATIC
  include/classifier/types.h
  include/classifier/metrics.h
  include/classifier/tokenizer.h
  include/classifier/vocabulary.h
  include/classifier/tfidf.h
  include/classifier/logistic_regression.h
  include/classifier/neural_classifier.h
  include/classifier/dataset.h
  include/classifier/model.h
  src/tokenizer.cpp
  src/metrics.cpp
  src/vocabulary.cpp
  src/tfidf.cpp
  src/logistic_regression.cpp
  src/neural_classifier.cpp
  src/dataset.cpp
  src/model.cpp
)

# Tests
add_executable(run_tests
  tests/test_main.cpp
  tests/test_tokenizer.cpp
  tests/test_math.cpp
  tests/test_tfidf.cpp
  tests/test_logistic.cpp
  tests/test_neural.cpp
  tests/test_model.cpp
)

target_link_libraries(run_tests PRIVATE classifier_core)

# Training & evaluation CLI
add_executable(train_cli
  apps/train_cli.cpp
)
target_link_libraries(train_cli PRIVATE classifier_core)
target_compile_definitions(train_cli PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
