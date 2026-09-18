cmake_minimum_required(VERSION 3.20)
project(classifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Determine if this project is being built standalone or included by a consumer
set(CLASSIFIER_IS_TOP_LEVEL OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CLASSIFIER_IS_TOP_LEVEL ON)
endif()

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

# Alias allows consumers to link `classifier::classifier_core`
add_library(classifier::classifier_core ALIAS classifier_core)

# Target-scoped include directories: consumers automatically get <classifier/...>
target_include_directories(classifier_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>$<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/3rdparty
)

# Apply strict warnings only when developing inside this repository
if(CLASSIFIER_IS_TOP_LEVEL)
  if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    target_compile_options(classifier_core PRIVATE /MP /GS /W4 /permissive-)
  else()
    target_compile_options(classifier_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  endif()

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
  target_compile_definitions(train_cli PRIVATE PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()