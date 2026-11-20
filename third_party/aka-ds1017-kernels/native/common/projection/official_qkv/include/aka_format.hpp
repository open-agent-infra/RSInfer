#pragma once
#include <cmath>
#include <string>
#if __has_include(<format>)
#include <format>
namespace aka_format {
using std::format;
inline std::string hex_literal(float v) { return std::format("{}0x{:a}f",std::signbit(v)?"-":"",std::abs(v)); }
}
#else
#define FMT_HEADER_ONLY
#include <fmt/format.h>
namespace aka_format {
using fmt::format;
inline std::string hex_literal(float v) { return fmt::format("{}{:a}f",std::signbit(v)?"-":"",std::abs(v)); }
}
#endif
