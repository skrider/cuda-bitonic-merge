// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND)                                   \
    {                                           \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    }                                           \
    else                                        \
    {                                           \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

// #define DTYPE_SWITCH(DTYPE, CONST_NAME, ...)       \
//   [&] {                                            \
//     if (DTYPE == BM_FLOAT16)                       \
//     {                                              \
//       constexpr static bool CONST_NAME = true;     \
//       using Element = cutlass::half_t;             \
//       return __VA_ARGS__();                        \
//     }                                              \
//     else if (DTYPE == BM_BFLOAT16)                 \
//     {                                              \
//       constexpr static bool CONST_NAME = true;     \
//       using Element = cutlass::bfloat16_t;         \
//       return __VA_ARGS__();                        \
//     }                                              \
//     else if (DTYPE == BM_INT16)                    \
//     {                                              \
//       constexpr static bool CONST_NAME = false;    \
//       using Element = cutlass::int16_t;            \
//       return __VA_ARGS__();                        \
//     }                                              \
//     else                                           \
//     {                                              \
//       TORCH_CHECK(false, "Unsupported data type"); \
//     }                                              \
//   }()
#define DTYPE_SWITCH(DTYPE, ...) \
  [&] {                          \
    if (DTYPE == BM_INT16)       \
    {                            \
      using Element = int16_t;   \
      return __VA_ARGS__();      \
    }                            \
  }()

#define SEQLEN_SWITCH(SEQLEN, ...)           \
  [&] {                                      \
    if (SEQLEN <= (1 << 8))                  \
    {                                        \
      constexpr static int blockN = 1 << 8;  \
      constexpr static int nWarps = 1;       \
      return __VA_ARGS__();                  \
    }                                        \
    else if (SEQLEN <= (1 << 9))             \
    {                                        \
      constexpr static int blockN = 1 << 9;  \
      constexpr static int nWarps = 2;       \
      return __VA_ARGS__();                  \
    }                                        \
    else if (SEQLEN <= (1 << 10))            \
    {                                        \
      constexpr static int blockN = 1 << 10; \
      constexpr static int nWarps = 4;       \
      return __VA_ARGS__();                  \
    }                                        \
    else if (SEQLEN <= (1 << 11))            \
    {                                        \
      constexpr static int blockN = 1 << 11; \
      constexpr static int nWarps = 8;       \
      return __VA_ARGS__();                  \
    }                                        \
    else                                     \
    {                                        \
      constexpr static int blockN = 1 << 12; \
      constexpr static int nWarps = 16;      \
      return __VA_ARGS__();                  \
    }                                        \
  }()
