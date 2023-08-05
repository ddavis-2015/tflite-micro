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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

constexpr int kSamplesPerMS = kAudioSampleFrequency / 1000;
constexpr size_t kFeatureSliceSamples = kFeatureSliceDurationMs * kSamplesPerMS;

void DumpFeatureData(const int8_t* slice_data) {
  for (int i = 0; i < kFeatureSliceSize; i += 8) {
    MicroPrintf("%3d, %3d, %3d, %3d, %3d, %3d, %3d, %3d,", slice_data[0],
                slice_data[1], slice_data[2], slice_data[3], slice_data[4],
                slice_data[5], slice_data[6], slice_data[7]);
    slice_data += 8;
  }
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MicroFeaturesGeneratorDumpYes) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, InitializeMicroFeatures());
  int8_t slice_data[kFeatureSliceSize] = {};
  size_t num_samples_processed = 0;

  TF_LITE_MICRO_EXPECT_EQ(kFeatureSliceSamples, g_yes_30ms_audio_data_size);
  TfLiteStatus status = GenerateMicroFeatures(
      g_yes_30ms_audio_data, kFeatureSliceSamples, kFeatureSliceSize,
      slice_data, &num_samples_processed);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);
  TF_LITE_MICRO_EXPECT_EQ(num_samples_processed, kFeatureSliceSamples);
  MicroPrintf("### BEGIN Feature Data: yes");
  DumpFeatureData(slice_data);
  MicroPrintf("### END Feature Data: yes");
}

TF_LITE_MICRO_TEST(MicroFeaturesGeneratorDumpNo) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, InitializeMicroFeatures());
  int8_t slice_data[kFeatureSliceSize] = {};
  size_t num_samples_processed = 0;

  TF_LITE_MICRO_EXPECT_EQ(kFeatureSliceSamples, g_no_30ms_audio_data_size);
  TfLiteStatus status = GenerateMicroFeatures(
      g_no_30ms_audio_data, kFeatureSliceSamples, kFeatureSliceSize, slice_data,
      &num_samples_processed);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);
  TF_LITE_MICRO_EXPECT_EQ(num_samples_processed, kFeatureSliceSamples);
  MicroPrintf("### BEGIN Feature Data: no");
  DumpFeatureData(slice_data);
  MicroPrintf("### END Feature Data: no");
}

TF_LITE_MICRO_TESTS_END
