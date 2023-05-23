// Copyright (c) 2023, DeepLink.
#pragma once

#include <stdio.h>
#include <iostream>

#define CONCAT_(prefix, suffix) prefix##suffix
#define CONCAT(prefix, suffix) CONCAT_(prefix, suffix)
#define MAKE_UNIQUE_VARIABLE_NAME(prefix) CONCAT(prefix##_, __LINE__)

#define DIPU_LOG std::cout << __FILE__ << ":" << __LINE__ << " "
#define DIPU_LOG_ONCE               \
    static auto& __attribute__((unused)) MAKE_UNIQUE_VARIABLE_NAME(__func__) = DIPU_LOG

#define DIPU_LOG_ERROR std::cerr << __FILE__ << ":" << __LINE__ << " "
#define DIPU_LOG_ERROR_ONCE             \
    static auto& __attribute__((unused)) MAKE_UNIQUE_VARIABLE_NAME(__func__) = DIPU_LOG_ERROR
