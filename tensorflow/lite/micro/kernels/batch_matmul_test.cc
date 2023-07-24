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

  static const TFLMRegistration registration = tflite::Register_BATCH_MATMUL();

  alignas(micro::KernelRunner) static char
      kernel_runner_buffer[sizeof(micro::KernelRunner)] = {};
  MicroPrintf("kernel_runner_buffer = %p", kernel_runner_buffer);

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
  MicroPrintf("invoke result before: %d, runner = %p",
              micro_test::did_test_fail, runner);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->Invoke());
  MicroPrintf("invoke result after: %d, runner = %p", micro_test::did_test_fail,
              runner);

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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Flatten) {
  int kInputDims_LHS[] = {4, 3, 2, 2, 4};
  int kInputDims_RHS[] = {4, 3, 1, 4, 1};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 48;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 12;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 1);

  constexpr float kExpect[] = {30,  70,  110,  150,  486,  590,
                               694, 798, 1454, 1622, 1790, 1958};
  int kOutputDims[] = {4, 3, 2, 2, 1};
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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Simple) {
  int kInputDims_LHS[] = {3, 1, 2, 3};
  int kInputDims_RHS[] = {3, 1, 3, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 6;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 12;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {74., 80., 86., 92., 173., 188., 203., 218.};
  int kOutputDims[] = {3, 1, 2, 4};
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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_SimpleRHSAdjoint) {
  int kInputDims_LHS[] = {3, 1, 2, 3};
  int kInputDims_RHS[] = {3, 1, 4, 3};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 6;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr float kInput_RHS[] = {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18};

  constexpr float kExpect[] = {74., 80., 86., 92., 173., 188., 203., 218.};
  int kOutputDims[] = {3, 1, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      true,   // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_SimpleLHSAdjoint) {
  int kInputDims_LHS[] = {3, 1, 3, 2};
  int kInputDims_RHS[] = {3, 1, 3, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};
  constexpr float kInput_LHS[] = {1, 4, 2, 5, 3, 6};

  constexpr size_t kInputSize_RHS = 12;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {74., 80., 86., 92., 173., 188., 203., 218.};
  int kOutputDims[] = {3, 1, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,   // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BatchSizeTwo) {
  int kInputDims_LHS[] = {3, 2, 2, 3};
  int kInputDims_RHS[] = {3, 2, 3, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};
  constexpr size_t kInputSize_LHS = 12;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 24;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {74.,  80.,  86.,  92.,  173., 188., 203., 218.,
                               560., 584., 608., 632., 767., 800., 833., 866.};
  int kOutputDims[] = {3, 2, 2, 4};
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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast) {
  int kInputDims_LHS[] = {3, 2, 2, 3};
  int kInputDims_RHS[] = {2, 3, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};
  constexpr size_t kInputSize_LHS = 12;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 12;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {74.,  80.,  86.,  92.,  173., 188., 203., 218.,
                               272., 296., 320., 344., 371., 404., 437., 470.};
  int kOutputDims[] = {3, 2, 2, 4};
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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BroadcastLHSAdjoint) {
  int kInputDims_LHS[] = {3, 2, 3, 2};
  int kInputDims_RHS[] = {2, 3, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr float kInput_LHS[] = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};

  constexpr size_t kInputSize_RHS = 12;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {74.,  80.,  86.,  92.,  173., 188., 203., 218.,
                               272., 296., 320., 344., 371., 404., 437., 470.};
  int kOutputDims[] = {3, 2, 2, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,   // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2) {
  int kInputDims_LHS[] = {4, 2, 1, 3, 2};
  int kInputDims_RHS[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 12;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 24;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  int kOutputDims[] = {4, 2, 3, 3, 4};
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

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2LHSAdjoint) {
  int kInputDims_LHS[] = {4, 2, 1, 2, 3};
  int kInputDims_RHS[] = {3, 3, 2, 4};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr float kInput_LHS[] = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};

  constexpr size_t kInputSize_RHS = 24;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,   // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2RHSAdjoint) {
  int kInputDims_LHS[] = {4, 2, 1, 3, 2};
  int kInputDims_RHS[] = {3, 3, 4, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 12;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr float kInput_RHS[] = {7,  11, 8,  12, 9,  13, 10, 14,
                                  15, 19, 16, 20, 17, 21, 18, 22,
                                  23, 27, 24, 28, 25, 29, 26, 30};

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      true,   // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2BothAdjoint) {
  int kInputDims_LHS[] = {4, 2, 1, 2, 3};
  int kInputDims_RHS[] = {3, 3, 4, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr float kInput_LHS[] = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};

  constexpr float kInput_RHS[] = {7,  11, 8,  12, 9,  13, 10, 14,
                                  15, 19, 16, 20, 17, 21, 18, 22,
                                  23, 27, 24, 28, 25, 29, 26, 30};

  constexpr float kExpect[] = {
      29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101., 112., 123., 134.,
      53.,  56.,  59.,  62.,  121., 128., 135., 142., 189., 200., 211., 222.,
      77.,  80.,  83.,  86.,  177., 184., 191., 198., 277., 288., 299., 310.,
      137., 152., 167., 182., 173., 192., 211., 230., 209., 232., 255., 278.,
      257., 272., 287., 302., 325., 344., 363., 382., 393., 416., 439., 462.,
      377., 392., 407., 422., 477., 496., 515., 534., 577., 600., 623., 646.};
  int kOutputDims[] = {4, 2, 3, 3, 4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      true,  // adj_x
      true,  // adj_y
      false  // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data);
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BroadcastFromRHS) {
  int kInputDims_LHS[] = {2, 4, 5};
  int kInputDims_RHS[] = {4, 3, 1, 5, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr size_t kInputSize_LHS = 20;
  float kInput_LHS[kInputSize_LHS];
  std::iota(std::begin(kInput_LHS), std::end(kInput_LHS), 1);

  constexpr size_t kInputSize_RHS = 30;
  float kInput_RHS[kInputSize_RHS];
  std::iota(std::begin(kInput_RHS), std::end(kInput_RHS), 7);

  constexpr float kExpect[] = {185.,  200.,  460.,  500.,  735.,  800.,
                               1010., 1100., 335.,  350.,  860.,  900.,
                               1385., 1450., 1910., 2000., 485.,  500.,
                               1260., 1300., 2035., 2100., 2810., 2900.};
  int kOutputDims[] = {4, 3, 1, 4, 2};
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

TF_LITE_MICRO_TEST(ConstRHSBatchMatMulOpModelRHSNotAdjoint) {
  int kInputDims_LHS[] = {3, 1, 6, 2};
  int kInputDims_RHS[] = {2, 2, 3};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_LHS,
                                                  kInputDims_RHS};

  constexpr float kInput_LHS[] = {6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7};

  constexpr float kInput_RHS[] = {6, 3, 7, 4, 6, 9};

  constexpr float kExpect[] = {48, 36, 69, 58, 45, 85, 72, 72, 123,
                               36, 42, 68, 58, 45, 85, 46, 51, 84};
  int kOutputDims[] = {3, 1, 6, 3};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  constexpr TfLiteBatchMatMulParams params = {
      false,  // adj_x
      false,  // adj_y
      false   // asymmetric_quantize_inputs
  };

  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data, true);
  // Eval twice to make sure constant transposed RHS is persistent.
  tflite::testing::TestBatchMatMulFloat(params, kInputDims, kInput_LHS,
                                        kInput_RHS, kOutputDims, kExpect,
                                        output_data, true, false);
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

#endif  // notdef

TF_LITE_MICRO_TESTS_END
