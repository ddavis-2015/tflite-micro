/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kTestTolerance = 1e-5;
constexpr int kNumInputs = 2;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex_LHS = 0;
constexpr int kInputTensorIndex_RHS = 1;
constexpr int kOutputTensorIndex = 2;

// min/max are used to compute symmetric scale, zero-point is 0
// scale should be 0 to use min/max
template <typename T, size_t kInputSize>
struct TestQuantizationParams {
  // quantization parameters
  float scale;  // if 0, use data_min and data_max
  int32_t zero_point;
  float data_min;  // input data minimum value
  float data_max;  // input data maximum value

  T input_data[kInputSize];  // quantized input storage
};

micro::KernelRunner* GetKernelRunnerInstance(
    TfLiteTensor* tensors, int tensors_count,
    const TfLiteBatchMatMulParams& params, bool need_init_prepare) {
  static int kInputArrayData[] = {kNumInputs, kInputTensorIndex_LHS,
                                  kInputTensorIndex_RHS};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  static int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TFLMRegistration registration = tflite::Register_BATCH_MATMUL();

  alignas(micro::KernelRunner) static char
      kernel_runner_buffer[sizeof(micro::KernelRunner)];

  static micro::KernelRunner* runner = nullptr;
  if (runner == nullptr || need_init_prepare) {
    runner = new (kernel_runner_buffer) micro::KernelRunner(
        registration, tensors, tensors_count, inputs_array, outputs_array,
        const_cast<TfLiteBatchMatMulParams*>(&params));

    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->InitAndPrepare());
    MicroPrintf("init/prepare result: %d", micro_test::did_test_fail);
  }

  return runner;
}

#ifdef notyet
template <typename T, size_t N>
void TestBatchMatMulQuantized(TestBatchMatMulParams<T, N>& params,
                              int* input_dims_data[kNumInputs],
                              const int32_t* input_data_lhs,
                              const float* input_data_rhs, int* expected_dims,
                              const float* expected_data, float* output_data) {
  TfLiteIntArray* input_dims_lhs = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_rhs = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  const float scale =
      SymmetricScaleFromMinMax<int8_t>(params.data_min, params.data_max);

  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data_rhs, params.input_data, input_dims_rhs,
                            scale, 0),
      CreateTensor(input_data_lhs, input_dims_lhs),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  micro::KernelRunner* runner =
      InitAndPrepareKernelRunner(tensors, tensors_count);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTestTolerance);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}
#endif  // notyet

void TestBatchMatMulFloat(const TfLiteBatchMatMulParams& params,
                          int* input_dims_data[kNumInputs],
                          const float* input_data_lhs,
                          const float* input_data_rhs, int* expected_dims,
                          const float* expected_data, float* output_data,
                          bool need_constant_rhs = false,
                          bool need_init_prepare = true) {
  TfLiteIntArray* input_dims_lhs = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_rhs = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  static TfLiteTensor tensors[kNumInputs + kNumOutputs];

  if (need_init_prepare) {
    tensors[kInputTensorIndex_LHS] =
        CreateTensor(input_data_lhs, input_dims_lhs);
    tensors[kInputTensorIndex_RHS] =
        CreateTensor(input_data_rhs, input_dims_rhs);
    if (need_constant_rhs) {
      tensors[kInputTensorIndex_RHS].allocation_type = kTfLiteMmapRo;
    }
    tensors[kOutputTensorIndex] = CreateTensor(output_data, output_dims);
  }

  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  micro::KernelRunner* runner = GetKernelRunnerInstance(
      tensors, tensors_count, params, need_init_prepare);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->Invoke());
  MicroPrintf("invoke result: %d", micro_test::did_test_fail);

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTestTolerance);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Ones) {
  int kInputDims_LHS[] = {4, 3, 2, 1, 4};
  int kInputDims_RHS[] = {4, 3, 1, 4, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 24;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 12;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 1);

  constexpr float kExpect[] = {30, 70, 278, 382, 782, 950};
  int kOutputDims[] = {4, 3, 2, 1, 1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

#ifdef notdef
TF_LITE_MICRO_TEST(HybridEmbeddingLookupHybridOpTestSimple2DTestInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {2, 3, 8};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {2, 3, 8};

  constexpr int32_t kInput_LHS[kInputSize_LHS] = {1, 0, 2};
  constexpr float kInput_RHS[] = {
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kInputCount_1 = std::extent<decltype(kInput_RHS)>::value;
  constexpr float kExpect[] = {
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestBatchMatMulParams<kInputCount_1> params = {};
  auto minmax =
      std::minmax_element(std::begin(kInput_RHS), std::end(kInput_RHS));
  params.data_max = *minmax.second;
  params.data_min = *minmax.first;

  tflite::testing::TestEmbeddingLookupQuantized(
      params, kInputDims, kInput_LHS, kInputSize_LHS kInput_RHS, kOutputDims,
      kExpect, output_data);
}

TF_LITE_MICRO_TEST(HybridEmbeddingLookupHybridOpTestSimple3DTestInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 2, 4};

  constexpr int32_t kInput_LHS[kInputSize_LHS] = {1, 0, 2};
  constexpr float kInput_RHS[] = {
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kInputCount_1 = std::extent<decltype(kInput_RHS)>::value;
  constexpr float kExpect[] = {
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestBatchMatMulParams<kInputCount_1> params = {};
  auto minmax =
      std::minmax_element(std::begin(kInput_RHS), std::end(kInput_RHS));
  params.data_max = *minmax.second;
  params.data_min = *minmax.first;

  tflite::testing::TestEmbeddingLookupQuantized(
      params, kInputDims, kInput_LHS, kInputSize_LHS kInput_RHS, kOutputDims,
      kExpect, output_data);
}

TF_LITE_MICRO_TEST(HybridEmbeddingLookupHybridOpTestSimple4DTestInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {4, 3, 2, 2, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {4, 3, 2, 2, 2};

  constexpr int32_t kInput_LHS[kInputSize_LHS] = {1, 0, 2};
  constexpr float kInput_RHS[] = {
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kInputCount_1 = std::extent<decltype(kInput_RHS)>::value;
  constexpr float kExpect[] = {
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestBatchMatMulParams<kInputCount_1> params = {};
  auto minmax =
      std::minmax_element(std::begin(kInput_RHS), std::end(kInput_RHS));
  params.data_max = *minmax.second;
  params.data_min = *minmax.first;

  tflite::testing::TestEmbeddingLookupQuantized(
      params, kInputDims, kInput_LHS, kInputSize_LHS kInput_RHS, kOutputDims,
      kExpect, output_data);
}

TF_LITE_MICRO_TEST(EmbeddingLookupOpTestSimpleInt8) {
  int kInputDims_0[] = {1, 3};
  int kInputDims_1[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {3, 3, 2, 4};

  constexpr int32_t kInput_LHS[kInputSize_LHS] = {1, 0, 2};
  constexpr int8_t kInput_RHS[] = {
      0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
      100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
      -56, -55, -54, -53, -46, -45, -44, -43,  // Row 2
  };
  constexpr int8_t kExpect[] = {
      100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
      0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
      -56, -55, -54, -53, -46, -45, -44, -43,  // Row 2
  };
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int8_t output_data[kOutputCount];

  tflite::testing::TestEmbeddingLookup(kInputDims, kInput_LHS,
                                       kInputSize_LHS kInput_RHS, kOutputDims,
                                       kExpect, output_data);
}
#endif  // notdef

TF_LITE_MICRO_TESTS_END
