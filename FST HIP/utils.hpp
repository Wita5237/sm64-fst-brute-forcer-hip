#include <string>

#pragma once
#include <hip/hip_runtime.h>


int getMinPrecision(float num);
std::string float2string(float num, int precision);
std::string float2string_max(float num);
std::string float2string_max(float num, int buffer);