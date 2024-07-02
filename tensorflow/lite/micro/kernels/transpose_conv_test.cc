/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kTolerance = 1e-5;

// Common inputs and outputs.
constexpr int kInputElements = 32;
static int kInputShape[] = {4, 1, 4, 4, 2};
static const float kInputData[kInputElements] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

constexpr int kFilterElements = 18;
static int kFilterShape[] = {4, 1, 3, 3, 2};
static const float kFilterData[kFilterElements] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

constexpr int kBiasElements = 1;
static int kBiasShape[] = {4, 1, 1, 1, 1};
static const float kBiasData[kBiasElements] = {0};

constexpr int kOutputElements = 16;
static int kOutputShape[] = {4, 1, 4, 4, 1};
static const float kGoldenData[kOutputElements] = {
    184,  412,  568,  528,  678,  1347, 1689, 1434,
    1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760};

// Transpose conv uses TfLiteConvParams.
static TfLiteConvParams common_conv_params = {kTfLitePaddingSame,  // padding
                                              1,  // stride_width
                                              1,  // stride_height
                                              kTfLiteActNone,
                                              1,
                                              1,
                                              kTfLiteNoType};

// Compression inputs and associated data
constexpr int kMaxTensors = 5;
constexpr int kFilterTensor = 1;  // physical index
constexpr int kBiasTensor = 3;    // physical index
constexpr int kOutputTensor = 2;  // physical index

template <typename TFILTER, typename TBIAS>
struct TestCompressionInfo {
  const TFILTER* filter_value_table;
  int filter_bit_width;
  bool use_filter_alt_axis;
  const TBIAS* bias_value_table;
  int bias_bit_width;
  bool is_bias_per_channel;
  bool use_bias_alt_axis;
  MicroContext::CompressionScheme scheme;
};

template <typename TBIAS>
struct TestCompressionQuantizedInfo : TestCompressionInfo<int8_t, TBIAS> {
  const uint8_t* filter_compressed;
  int8_t* filter_quantized;
  const int* filter_dims_data;    // TfLiteIntArray
  const float* filter_scales;     // TfLiteFloatArray
  const int* filter_zero_points;  // TfLiteIntArray

  const uint8_t* bias_compressed;
  TBIAS* bias_quantized;
  const int* bias_dims_data;  // TfLiteIntArray
  float* bias_scales;         // TfLiteFloatArray (computed)
  int* bias_zero_points;      // TfLiteIntArray (computed)
};

template <typename CTF = void, typename CTB = void>
TfLiteStatus InvokeTransposeConv(
    TfLiteTensor* tensors, int tensors_size,
    const TfLiteConvParams* conv_params,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr) {
  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  // TODO: account for optional bias tensor

  MicroContext::CompressionTensorData* compressed_tensors[kMaxTensors] = {};
  MicroContext::CompressionTensorData filter_comp_data = {};
  MicroContext::CompressionTensorData bias_comp_data = {};
  MicroContext::CompressedTensorList comp_list = {compressed_tensors};
  MicroContext::CompressedTensorList* comp_list_p = nullptr;

  if (comp_info != nullptr) {
    if (comp_info->scheme == MicroContext::CompressionScheme::kBinQuant) {
      bool is_per_channel_quantized =
          std::is_same<CTF, float>::value ? false : true;
      if (comp_info->filter_value_table != nullptr) {
        compressed_tensors[kFilterTensor] = &filter_comp_data;
        filter_comp_data.scheme = MicroContext::CompressionScheme::kBinQuant;
        filter_comp_data.data.bin_quant.compressed_bit_width =
            comp_info->filter_bit_width;
        filter_comp_data.data.bin_quant.value_table =
            comp_info->filter_value_table;
        filter_comp_data.data.bin_quant.is_per_channel_quantized =
            is_per_channel_quantized;
        filter_comp_data.data.bin_quant.use_alternate_axis =
            comp_info->use_filter_alt_axis;
      }
      if (comp_info->bias_value_table != nullptr) {
        compressed_tensors[kBiasTensor] = &bias_comp_data;
        bias_comp_data.scheme = MicroContext::CompressionScheme::kBinQuant;
        bias_comp_data.data.bin_quant.compressed_bit_width =
            comp_info->bias_bit_width;
        bias_comp_data.data.bin_quant.value_table = comp_info->bias_value_table;
        bias_comp_data.data.bin_quant.is_per_channel_quantized =
            is_per_channel_quantized;
        bias_comp_data.data.bin_quant.use_alternate_axis =
            comp_info->use_bias_alt_axis;
      }
      comp_list_p = &comp_list;
    } else {
      return kTfLiteError;
    }
  }

  const TFLMRegistration registration = tflite::Register_TRANSPOSE_CONV();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, conv_params, nullptr, comp_list_p);

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  return runner.Invoke();
}

template <typename T, typename CTF = void, typename CTB = void>
TfLiteStatus ValidateTransposeConvGoldens(
    TfLiteTensor* tensors, int tensors_size, const float* expected_output_data,
    int output_length, float* output_data, T* output_quantized,
    TfLiteConvParams* conv_params, float tolerance = kTolerance,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr) {
  TfLiteStatus status =
      InvokeTransposeConv(tensors, tensors_size, conv_params, comp_info);
  if (status != kTfLiteOk) {
    return status;
  }

  if (output_quantized != nullptr) {
    // TODO: account for optional bias tensor
    const float scale = tensors[kOutputTensor].params.scale;
    const int zero_point = tensors[kOutputTensor].params.zero_point;
    Dequantize(output_quantized, output_length, scale, zero_point, output_data);
    MicroPrintf("Dequantize: scale %f zero_point %d length %d", (double)scale,
                zero_point, output_length);
  }
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }

  return kTfLiteOk;
}

template <typename CTF = void, typename CTB = void>
TfLiteStatus TestTransposeConvFloat(
    int* input_dims_data, const float* input_data, int* filter_dims_data,
    const float* filter_data, int* bias_dims_data, const float* bias_data,
    int* output_dims_data, const float* expected_output_data,
    TfLiteConvParams* conv_params, float* output_data,
    const TestCompressionInfo<CTF, CTB>* comp_info = nullptr) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(filter_data, filter_dims),
      CreateTensor(input_data, input_dims),
      CreateTensor(bias_data, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  const int output_dims_count = ElementCount(*output_dims);
  return ValidateTransposeConvGoldens<float>(
      tensors, tensors_size, expected_output_data, output_dims_count,
      output_data, nullptr, conv_params, kTolerance, comp_info);
}

template <typename TBIAS, typename TIO>
TfLiteStatus TestTransposeConvQuantized(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_quantized, int* bias_dims_data,
    const float* bias_data, TBIAS* bias_quantized, int* output_dims_data,
    const float* expected_output_data, float* output_data,
    TIO* output_quantized, float output_scale, int output_zero_point,
    TfLiteConvParams* conv_params) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  int filter_zero_points[5];
  float filter_scales[std::extent<decltype(filter_zero_points)>::value];
  TfLiteAffineQuantization filter_quant;
  TF_LITE_MICRO_EXPECT_LE(static_cast<size_t>(filter_dims->data[0]),
                          std::extent<decltype(filter_zero_points)>::value - 1);
  TF_LITE_MICRO_CHECK_FAIL();
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */);

  int bias_zero_points[std::extent<decltype(filter_zero_points)>::value];
  float bias_scales[std::extent<decltype(filter_scales)>::value];
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor bias_tensor = {};
  if (filter_quant.scale->size > 1) {
    bias_tensor = CreatePerChannelQuantizedBiasTensor(
        bias_data, bias_quantized, bias_dims, input_scale, filter_scales,
        bias_scales, bias_zero_points, &bias_quant,
        0 /* quantized dimension */);
  } else {
    bias_tensor =
        CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                  input_scale, filter_quant.scale->data[0]);
    MicroPrintf("input scale %f filter scale %f filter size %d",
                (double)input_scale, (double)filter_quant.scale->data[0],
                filter_quant.scale->size);
  }

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      bias_tensor,
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  // Tolerance is slightly looser for 8x16 compared with float, since quant
  // error is more pronounced on the finer-grained 16-bit output.
  constexpr float tolerance = std::is_same<TIO, int8_t>::value ? 1.0f : 4.0f;
  const int output_dims_count = ElementCount(*output_dims);
  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_data, output_dims_count,
      output_data, output_quantized, conv_params, tolerance);
}

template <typename TIO, typename CTB>
TfLiteStatus TestTransposeConvQuantizedCompressed(
    int* input_dims_data, const float* input_data, TIO* input_quantized,
    float input_scale, int input_zero_point, int* output_dims_data,
    const float* expected_output_data, float* output_data,
    TIO* output_quantized, float output_scale, int output_zero_point,
    TfLiteConvParams* conv_params,
    const TestCompressionQuantizedInfo<CTB>* comp_info) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(comp_info->filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(comp_info->bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteFloatArray* filter_scales =
      FloatArrayFromFloats(comp_info->filter_scales);
  TfLiteIntArray* filter_zero_points =
      IntArrayFromInts(comp_info->filter_zero_points);
  TfLiteFloatArray* bias_scales = FloatArrayFromFloats(comp_info->bias_scales);
  TfLiteIntArray* bias_zero_points =
      IntArrayFromInts(comp_info->bias_zero_points);

  size_t quantized_axis;

  TfLiteAffineQuantization filter_quant_params;
  quantized_axis = comp_info->use_filter_alt_axis ? 3 : 0;
  TfLiteTensor filter_tensor = CreatePerChannelQuantizedTensor(
      comp_info->filter_compressed, filter_dims, filter_scales,
      filter_zero_points, &filter_quant_params, quantized_axis, false,
      kTfLiteInt8);

  TfLiteAffineQuantization bias_quant_params;
  quantized_axis = comp_info->use_bias_alt_axis ? 3 : 0;
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensorCompressed(
      comp_info->bias_value_table, comp_info->bias_quantized,
      comp_info->bias_compressed, bias_dims, input_scale, filter_scales,
      bias_scales, bias_zero_points, &bias_quant_params, quantized_axis);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  constexpr int tensors_size = kMaxTensors;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point),
      bias_tensor,
      CreateQuantizedTensor(output_quantized, output_dims, output_scale,
                            output_zero_point),
  };

  // Tolerance is slightly looser for 8x16 compared with float, since quant
  // error is more pronounced on the finer-grained 16-bit output.
  constexpr float tolerance = std::is_same<TIO, int8_t>::value ? 1.0f : 4.0f;
  const int output_dims_count = ElementCount(*output_dims);
  return ValidateTransposeConvGoldens(
      tensors, tensors_size, expected_output_data, output_dims_count,
      output_data, output_quantized, conv_params, tolerance, comp_info);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTestFloat) {
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          tflite::testing::kBiasShape, tflite::testing::kBiasData,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          &tflite::testing::common_conv_params, output_data));
}

TF_LITE_MICRO_TEST(SimpleTestFloatCompressed) {
  float output_data[tflite::testing::kOutputElements];

  // compressed filter data for kBinQuant scheme
  constexpr uint8_t kBinQuantFilterData[] = {
      0x00, 0x44, 0x32, 0x14, 0xC7, 0x42, 0x54, 0xB6, 0x35, 0xCF, 0x84, 0x40};
  constexpr int kBinQuantFilterBitWidth = 5;
  // compressed bias data for kBinQuant scheme
  constexpr uint8_t kBinQuantBiasData[] = {0x00};
  constexpr int kBinQuantBiasBitWidth = 1;

  tflite::testing::TestCompressionInfo<float, float> comp_info = {};
  comp_info.scheme = tflite::MicroContext::CompressionScheme::kBinQuant;
  comp_info.filter_value_table = tflite::testing::kFilterData;
  comp_info.filter_bit_width = kBinQuantFilterBitWidth;
  comp_info.bias_value_table = tflite::testing::kBiasData;
  comp_info.bias_bit_width = kBinQuantBiasBitWidth;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          tflite::testing::kFilterShape,
          reinterpret_cast<const float*>(kBinQuantFilterData),
          tflite::testing::kBiasShape,
          reinterpret_cast<const float*>(kBinQuantBiasData),
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          &tflite::testing::common_conv_params, output_data, &comp_info));
}

TF_LITE_MICRO_TEST(fusedRELUTest) {
  float output_data[tflite::testing::kOutputElements];
  float golden_data[] = {29,  24,  0, 0, 99,  72,  0,   0,
                         207, 186, 0, 0, 263, 292, 141, 0};
  int filter_shape[] = {4, 1, 3, 3, 1};
  float filter_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int input_shape[] = {4, 1, 4, 4, 1};
  float input_data[] = {1, 2,  -3,  -4,  5,  6,  -7, -8,
                        9, 10, -11, -12, 13, 14, 15, 16};
  TfLiteConvParams conv_params = {kTfLitePaddingSame,  // padding
                                  1,                   // stride_width
                                  1,                   // stride_height
                                  kTfLiteActRelu,
                                  1,
                                  1,
                                  kTfLiteNoType};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestTransposeConvFloat(
                     input_shape, input_data, filter_shape, filter_data,
                     tflite::testing::kBiasShape, tflite::testing::kBiasData,
                     tflite::testing::kOutputShape, golden_data, &conv_params,
                     output_data));
}

TF_LITE_MICRO_TEST(AccuracyWithFusedActivationTest) {
  int output_shape[] = {4, 1, 3, 4, 1};
  float output_data[tflite::testing::kOutputElements];
  float golden_data[] = {1615, 1938, 0, 0, 2584, 1615, 0, 0, 323, 1292, 0, 0};
  int filter_shape[] = {4, 1, 3, 3, 1};
  float filter_data[] = {9, 5, 6, 9, 8, 5, 3, 1, 4};
  int input_shape[] = {4, 1, 1, 2, 1};
  float input_data[] = {323, -521};
  TfLiteConvParams conv_params = {kTfLitePaddingSame,  // padding
                                  3,                   // stride_width
                                  3,                   // stride_height
                                  kTfLiteActRelu,
                                  1,
                                  1,
                                  kTfLiteNoType};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::TestTransposeConvFloat(
                     input_shape, input_data, filter_shape, filter_data,
                     tflite::testing::kBiasShape, tflite::testing::kBiasData,
                     output_shape, golden_data, &conv_params, output_data));
}

TF_LITE_MICRO_TEST(MultiChannelBiasWithFusedActivationTest) {
  int output_shape[] = {4, 1, 5, 5, 2};
  float output_data[50];
  float golden_data[] = {4,  6,  6,  8,  10, 14, 9,  12, 13, 16, 10, 12, 12,
                         14, 28, 32, 21, 24, 25, 28, 13, 12, 9,  8,  35, 40,
                         45, 52, 57, 64, 0,  0,  0,  0,  0,  0,  39, 44, 47,
                         52, 0,  0,  0,  0,  4,  6,  63, 68, 71, 76};
  int filter_shape[] = {4, 2, 3, 3, 1};
  float filter_data[] = {1, 3, 5, 7, 9,  11, 13, 15, 17,
                         2, 4, 6, 8, 10, 12, 14, 16, 18};
  int input_shape[] = {4, 1, 2, 2, 1};
  float input_data[] = {1, 2, -3, 4};
  int bias_shape[] = {4, 2, 1, 1, 1};
  float bias_data[] = {3, 4};
  TfLiteConvParams conv_params = {kTfLitePaddingValid,  // padding
                                  2,                    // stride_width
                                  2,                    // stride_height
                                  kTfLiteActRelu,
                                  1,
                                  1,
                                  kTfLiteNoType};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvFloat(
          input_shape, input_data, filter_shape, filter_data, bias_shape,
          bias_data, output_shape, golden_data, &conv_params, output_data));
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannel) {
  const float input_scale = 1.0f;   // 0.5f;
  const float output_scale = 1.0f;  // 30.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

#ifdef notdef
TEST_P(TransposeConvOpTest, SimpleTestQuantizedPerChannelSingleChannel) {
  const std::initializer_list<float> filter_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::initializer_list<int8_t> const_filter_data = {14, 28, 42,  56, 71,
                                                           85, 99, 113, 127};
  PerChannelQuantizedTransposeConvOpModel model(
      GetRegistration(), {1, 4, 4, 1},
      {TensorType_INT8, {1, 3, 3, 1}, 0, 0, 0, 0, true, {9.0 / 127}, {0}, 0},
      const_filter_data,
      {TensorType_INT8, {1, 4, 4, 1}, 0, 0, 16.0 / 127, -128},
      {TensorType_INT8, {}, 0, 0, 2, -128}, Padding_SAME, 1, 1,
      ActivationFunctionType_NONE, GetTestType(),
      /* version */ 2);
  model.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  if (GetTestType() == TestType::kDynamic) {
    model.SetFilter(filter_data);
  }
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      model.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({30, 62, 84, 76, 100, 192, 238, 198, 206,
                                       372, 416, 330, 262, 446, 484, 366},
                                      1e-5)));

  // GetOutputShape() should always be same as model.SetOutputShape(...);
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}
#endif

#ifdef notyet
TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelSingleChannelCompressed) {
  // data from TfLite test: SimpleTestQuantizedPerChannelSingleChannel
  const float input_scale = 0.5f;
  const float output_scale = 30.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  // compressed filter data for kBinQuant scheme
  constexpr uint8_t kBinQuantFilterData[] = {
      0x00, 0x44, 0x32, 0x14, 0xC7, 0x42, 0x54, 0xB6, 0x35, 0xCF, 0x84, 0x40};
  constexpr int kBinQuantFilterBitWidth = 5;
  // compressed bias data for kBinQuant scheme
  constexpr uint8_t kBinQuantBiasData[] = {0x00};
  constexpr int kBinQuantBiasBitWidth = 1;

  tflite::testing::TestCompressionQuantizedInfo<int32_t> comp_info = {};
  comp_info.scheme = tflite::MicroContext::CompressionScheme::kBinQuant;
  comp_info.filter_value_table = tflite::testing::kFilterData;
  comp_info.filter_bit_width = kBinQuantFilterBitWidth;
  comp_info.bias_value_table = tflite::testing::kBiasData;
  comp_info.bias_bit_width = kBinQuantBiasBitWidth;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantizedCompressed(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params, &comp_info));
}
#endif

TF_LITE_MICRO_TEST(SimpleTestQuantized16x8PerChannel) {
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int64_t bias_quantized[tflite::testing::kBiasElements];
  int16_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TEST(SimpleTestQuantized16x8PerChannelWithInt16Bias) {
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int16_t bias_quantized[tflite::testing::kBiasElements];
  int16_t output_quantized[tflite::testing::kOutputElements];
  float output_data[tflite::testing::kOutputElements];

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestTransposeConvQuantized(
          tflite::testing::kInputShape, tflite::testing::kInputData,
          input_quantized, input_scale, input_zero_point,
          tflite::testing::kFilterShape, tflite::testing::kFilterData,
          filter_quantized, tflite::testing::kBiasShape,
          tflite::testing::kBiasData, bias_quantized,
          tflite::testing::kOutputShape, tflite::testing::kGoldenData,
          output_data, output_quantized, output_scale, output_zero_point,
          &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TEST(InputOutputDifferentTypeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t output_data[tflite::testing::kOutputElements];

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateTensor(tflite::testing::kFilterData, filter_dims),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
  };
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError,
      tflite::testing::InvokeTransposeConv(
          tensors, tensors_size, &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TEST(HybridModeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);

  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t filter_data[tflite::testing::kFilterElements] = {};
  float output_data[tflite::testing::kOutputElements];

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims = IntArrayFromInts(output_shape_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateQuantizedTensor(filter_data, filter_dims,
                            /*scale=*/1.0f,
                            /*zero_point=*/0),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError,
      tflite::testing::InvokeTransposeConv(
          tensors, tensors_size, &tflite::testing::common_conv_params));
}

TF_LITE_MICRO_TESTS_END
