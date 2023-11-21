#!/usr/bin/env bash

# format all C/C++ files in current git repository with clang-format
git ls-files | grep -E "\.(c|h|cpp|cc|cxx|hpp|hh|hxx|cu|cuh)$" | xargs clang-format -i --style=file
