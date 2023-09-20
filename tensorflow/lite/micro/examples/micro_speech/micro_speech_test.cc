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
#include <cstdint>
#include <cstring>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kArenaSize = 28580;  // xtensa p6
uint8_t g_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

constexpr size_t kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr size_t kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

TfLiteStatus RegisterOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}

TfLiteStatus LoadMicroSpeechModelAndPerformInference(
    const Features& features, const char* expected_label) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MicroSpeechOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  MicroPrintf("MicroSpeech model arena size = %u",
              interpreter.arena_used_bytes());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  // check input shape is compatible with our feature data size
  TFLITE_CHECK_EQ(kFeatureElementCount,
                  input->dims->data[input->dims->size - 1]);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);
  // check output shape is compatible with our number of prediction categories
  TFLITE_CHECK_EQ(kCategoryCount, output->dims->data[output->dims->size - 1]);

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  std::copy_n(&features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(input));
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  // Dequantize output values
  float category_predictions[kCategoryCount];
  for (int i = 0; i < kCategoryCount; i++) {
    category_predictions[i] =
        (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
        output_scale;
  }
  int prediction_index =
      std::distance(std::begin(category_predictions),
                    std::max_element(std::begin(category_predictions),
                                     std::end(category_predictions)));
  TFLITE_CHECK(strcmp(expected_label, kCategoryLabels[prediction_index]) == 0);

  return kTfLiteOk;
}

TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const size_t audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  TFLITE_CHECK_NE(input, nullptr);
  // check input shape is compatible with our audio sample size
  TFLITE_CHECK_EQ(kAudioSampleDurationCount, audio_data_size);
  TFLITE_CHECK_EQ(kAudioSampleDurationCount,
                  input->dims->data[input->dims->size - 1]);

  TfLiteTensor* output = interpreter->output(0);
  TFLITE_CHECK_NE(output, nullptr);
  // check output shape is compatible with our feature size
  TFLITE_CHECK_EQ(kFeatureSize, output->dims->data[output->dims->size - 1]);

  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));
  TF_LITE_ENSURE_STATUS(interpreter->Invoke());
  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
              feature_output);

  return kTfLiteOk;
}

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  AudioPreprocessorOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  MicroPrintf("AudioPreprocessor model arena size = %u",
              interpreter.arena_used_bytes());

  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
    TF_LITE_ENSURE_STATUS(
        GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index], &interpreter));
    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  return kTfLiteOk;
}

TfLiteStatus TestAudioSample(const char* label, const int16_t* audio_data,
                             const size_t audio_data_size) {
  TF_LITE_ENSURE_STATUS(
      GenerateFeatures(audio_data, audio_data_size, &g_features));
  TF_LITE_ENSURE_STATUS(
      LoadMicroSpeechModelAndPerformInference(g_features, label));
  return kTfLiteOk;
}

}  // namespace

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  TF_LITE_ENSURE_STATUS(TestAudioSample("no", g_no_1000ms_audio_data,
                                        g_no_1000ms_audio_data_size));
  TF_LITE_ENSURE_STATUS(TestAudioSample("yes", g_yes_1000ms_audio_data,
                                        g_yes_1000ms_audio_data_size));
  TF_LITE_ENSURE_STATUS(TestAudioSample("silence", g_silence_1000ms_audio_data,
                                        g_silence_1000ms_audio_data_size));
  TF_LITE_ENSURE_STATUS(TestAudioSample("silence", g_noise_1000ms_audio_data,
                                        g_noise_1000ms_audio_data_size));
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
