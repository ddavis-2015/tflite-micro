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

#include <stdio.h>

#include <type_traits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/microfrontend/lib/bits.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

FrontendState g_micro_features_state;

template <typename T>
void DumpData(const T* datum, size_t datum_length, const char* name,
              const char* detail) {
  constexpr size_t kMaxLineLength = 80;
  size_t remain_line_length = kMaxLineLength;
  size_t item_length = 0;
  const char* item_format = "";

  if (std::is_same<T, int8_t>::value) {
    item_length = 6;
    item_format = "%3d, ";
  } else if (std::is_same<T, uint8_t>::value) {
    item_length = 5;
    item_format = "%3u, ";
  } else if (std::is_same<T, int16_t>::value) {
    item_length = 8;
    item_format = "%5d, ";
  } else if (std::is_same<T, uint16_t>::value) {
    item_length = 7;
    item_format = "%5u, ";
  } else if (std::is_same<T, int32_t>::value) {
    item_length = 13;
    item_format = "%10d, ";
  } else if (std::is_same<T, uint32_t>::value) {
    item_length = 12;
    item_format = "%10u, ";
  } else if (std::is_same<T, int64_t>::value) {
    item_length = 23;
    item_format = "%20lld, ";
  } else if (std::is_same<T, uint64_t>::value) {
    item_length = 22;
    item_format = "%20llu, ";
  } else if (std::is_same<T, float>::value) {
    item_length = 14;
    item_format = "%8.3f, ";
  } else if (std::is_same<T, complex_int16_t>::value) {
    item_length = 16;
    item_format = "%5d %5di, ";
  } else {
    printf("unknown data type\n");
    return;
  }

  printf("### BEGIN %s Data: %s\n", name, detail);

  for (size_t i = 0; i < datum_length; i++) {
    if (item_length > remain_line_length) {
      printf("\n");
      remain_line_length = kMaxLineLength;
    }
    remain_line_length -= item_length;
    if (std::is_same<T, complex_int16_t>::value) {
      const complex_int16_t* complex =
          reinterpret_cast<const complex_int16_t*>(datum) + i;
      printf(item_format, complex->real, complex->imag);
    } else {
      printf(item_format, datum[i]);
    }
  }
  if (remain_line_length != kMaxLineLength) {
    printf("\n");
  }
  printf("### END %s Data: %s\n", name, detail);
  fflush(stdout);
}

TfLiteStatus _InitializeMicroFeatures() {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 0;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    MicroPrintf("FrontendPopulateState() failed");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

struct FrontendOutput _FrontendProcessSamples(struct FrontendState* state,
                                              const int16_t* samples,
                                              size_t num_samples,
                                              size_t* num_samples_read,
                                              const char* detail) {
  DumpData(samples, num_samples, "Audio Frame", detail);
  DumpData(state->window.coefficients, num_samples, "Window Weights", detail);                                        

  struct FrontendOutput output;
  output.values = NULL;
  output.size = 0;

  // Try to apply the window - if it fails, return and wait for more data.
  if (!WindowProcessSamples(&state->window, samples, num_samples,
                            num_samples_read)) {
    return output;
  }
  DumpData(state->window.output, num_samples, "Window", detail);

  // Apply the FFT to the window's output (and scale it so that the fixed
  // point FFT can have as much resolution as possible).
  int input_shift =
      15 - MostSignificantBit32(state->window.max_abs_output_value);
  fprintf(stderr, "input shift %d [%s]\n", input_shift, detail);
  FftCompute(&state->fft, state->window.output, input_shift);
  DumpData(reinterpret_cast<int16_t*>(state->fft.output),
           state->fft.fft_size + 2, "FFT", detail);

  // We can re-ruse the fft's output buffer to hold the energy.
  int32_t* energy = (int32_t*)state->fft.output;

  FilterbankConvertFftComplexToEnergy(&state->filterbank, state->fft.output,
                                      energy);
  fprintf(stderr, "start index %d, end index %d [%s]\n",
          state->filterbank.start_index, state->filterbank.end_index, detail);
  DumpData(energy, state->fft.fft_size / 2 + 1, "Energy", detail);

  FilterbankAccumulateChannels(&state->filterbank, energy);
  DumpData(state->filterbank.work, state->filterbank.num_channels + 1,
           "Filterbank", detail);

  uint32_t* scaled_filterbank = FilterbankSqrt(&state->filterbank, input_shift);
  DumpData(scaled_filterbank, state->filterbank.num_channels,
           "Scaled Filterbank", detail);

  // Apply noise reduction.
  NoiseReductionApply(&state->noise_reduction, scaled_filterbank);
  DumpData(scaled_filterbank, state->noise_reduction.num_channels, "Noise",
           detail);

  if (state->pcan_gain_control.enable_pcan) {
    PcanGainControlApply(&state->pcan_gain_control, scaled_filterbank);
    DumpData(scaled_filterbank, state->pcan_gain_control.num_channels,
             "AGC Noise", detail);
  }

  // Apply the log and scale.
  int correction_bits =
      MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  fprintf(stderr, "correction bits %d [%s]\n", correction_bits, detail);
  uint16_t* logged_filterbank =
      LogScaleApply(&state->log_scale, scaled_filterbank,
                    state->filterbank.num_channels, correction_bits);
  DumpData(logged_filterbank, state->filterbank.num_channels, "Scaled Noise",
           detail);

  output.size = state->filterbank.num_channels;
  output.values = logged_filterbank;
  return output;
}

TfLiteStatus _GenerateMicroFeatures(const int16_t* input, int input_size,
                                    int output_size, int8_t* output,
                                    size_t* num_samples_read,
                                    const char* detail) {
  FrontendOutput frontend_output = _FrontendProcessSamples(
      &g_micro_features_state, input, input_size, num_samples_read, detail);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    // These scaling values are derived from those used in input_data.py in
    // the training pipeline. The feature pipeline outputs 16-bit signed
    // integers in roughly a 0 to 670 range. In training, these are then
    // arbitrarily divided by 25.6 to get float values in the rough range of
    // 0.0 to 26.0. This scaling is performed for historical reasons, to match
    // up with the output of other feature generators. The process is then
    // further complicated when we quantize the model. This means we have to
    // scale the 0.0 to 26.0 real values to the -128 to 127 signed integer
    // numbers. All this means that to get matching values from our integer
    // feature output into the tensor input, we have to perform: input =
    // (((feature / 25.6) / 26.0) * 256) - 128 To simplify this and perform it
    // in 32-bit integer math, we rearrange to: input = (feature * 256) /
    // (25.6 * 26.0) - 128
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    int32_t value =
        ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
        value_div;
    value -= 128;
    if (value < -128) {
      value = -128;
    }
    if (value > 127) {
      value = 127;
    }
    output[i] = value;
  }

  return kTfLiteOk;
}

constexpr int kSamplesPerMS = kAudioSampleFrequency / 1000;
constexpr size_t kFeatureSliceSamples = kFeatureSliceDurationMs * kSamplesPerMS;

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MicroFeaturesGeneratorDumpYes) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, _InitializeMicroFeatures());
  int8_t slice_data[kFeatureSliceSize] = {};
  size_t num_samples_processed = 0;

  TF_LITE_MICRO_EXPECT_EQ(kFeatureSliceSamples, g_yes_30ms_audio_data_size);
  TfLiteStatus status = _GenerateMicroFeatures(
      g_yes_30ms_audio_data, kFeatureSliceSamples, kFeatureSliceSize,
      slice_data, &num_samples_processed, "yes");
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);
  TF_LITE_MICRO_EXPECT_EQ(num_samples_processed, kFeatureSliceSamples);
  DumpData(slice_data, std::extent<decltype(slice_data)>::value, "Feature",
           "yes");
}

TF_LITE_MICRO_TEST(MicroFeaturesGeneratorDumpNo) {
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, _InitializeMicroFeatures());
  int8_t slice_data[kFeatureSliceSize] = {};
  size_t num_samples_processed = 0;

  TF_LITE_MICRO_EXPECT_EQ(kFeatureSliceSamples, g_no_30ms_audio_data_size);
  TfLiteStatus status = _GenerateMicroFeatures(
      g_no_30ms_audio_data, kFeatureSliceSamples, kFeatureSliceSize, slice_data,
      &num_samples_processed, "no");
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, status);
  TF_LITE_MICRO_EXPECT_EQ(num_samples_processed, kFeatureSliceSamples);
  DumpData(slice_data, std::extent<decltype(slice_data)>::value, "Feature",
           "no");
}

TF_LITE_MICRO_TESTS_END
