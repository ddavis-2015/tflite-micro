# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Audio Sample Preprocessor
"""

from __future__ import annotations
from typing import Callable, Any
from pathlib import Path
from dataclasses import dataclass

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import window_op
from tflite_micro.python.tflite_micro.signal.ops import fft_ops
from tflite_micro.python.tflite_micro.signal.ops import energy_op
from tflite_micro.python.tflite_micro.signal.ops import filter_bank_ops
from tflite_micro.python.tflite_micro.signal.ops import pcan_op
from tflite_micro.python.tflite_micro import runtime


_ENABLE_DEBUG = flags.DEFINE_enum(
    'debug_mode',
    'off',
    ['off', 'all', 'single'],
    'Enable debug output',
)

_FILE_TO_TEST = flags.DEFINE_enum(
    'file_to_test',
    'no',
    ['no', 'yes'],
    'File to test'
)


def _debug_print(*args):
  if _ENABLE_DEBUG.value != 'off':
    print(*args)


class _GenerateFeature(tf.Module):
  """Generate feature tensor from audio window samples"""

  def __init__(self, name: str, params: FeatureParams, detail: str):
    super().__init__(name=name)

    self._params = params
    window_sample_count: int = int(
        params.window_size_ms * params.sample_rate / 1000)
    hann_window_weights = window_op.hann_window_weights(
        window_sample_count, params.window_scaling_bits)
    self._hann_window_weights_tensor = tf.constant(
        hann_window_weights, name='hann_window_weights')
    self._fft_size, self._fft_size_log2 = fft_ops.get_pow2_fft_length(
        window_sample_count)
    self._filter_bank_index_start, self._filter_bank_index_end = \
        filter_bank_ops.calc_start_end_indices(
            self._fft_size,
            params.sample_rate,
            params.filter_bank_number_of_channels,
            params.filter_bank_lower_band_limit_hz,
            params.filter_bank_upper_band_limit_hz)
    self._debug_single = False
    self._detail = detail

  def generate_feature_for_frame(self, audio_frame: tf.Tensor) -> tf.Tensor:
    def _debug_print_internal(*args):
      if _ENABLE_DEBUG.value != 'off':
        if _ENABLE_DEBUG.value == 'single' and self._debug_single:
          return
        print(*args)

    _debug_print('*** generate_feature_for_frame ***')
    params = self._params
    detail = self._detail

    # update filter_bank_ops constants
    filter_bank_ops.FILTER_BANK_WEIGHT_SCALING_BITS = \
        params.filter_bank_scaling_bits
    filter_bank_ops.FILTER_BANK_ALIGNMENT = params.filter_bank_alignment
    filter_bank_ops.FILTER_BANK_CHANNEL_BLOCK_SIZE = \
        params.filter_bank_channel_block_size

    _debug_print_internal(f'audio frame output [{detail}]: {audio_frame!r}')

    # apply window to audio frame
    weights = self._hann_window_weights_tensor
    _debug_print_internal(f'window weights output [{detail}]: {weights!r}')
    window_output: tf.Tensor = window_op.window(
        audio_frame, weights, params.window_scaling_bits)
    _debug_print_internal(f'window output [{detail}]: {window_output!r}')

    # pre-scale window output
    window_output = tf.reshape(window_output, [-1])
    window_scaled_output, scaling_shift = fft_ops.fft_auto_scale(window_output)
    _debug_print_internal(f'scaling shift [{detail}]: {scaling_shift!r}')

    # compute FFT on scaled window output
    _debug_print_internal(
        f'fft size, log2 [{detail}]: {self._fft_size}, {self._fft_size_log2}')
    fft_output: tf.Tensor = fft_ops.rfft(window_scaled_output, self._fft_size)
    _debug_print_internal(f'fft output [{detail}]: {fft_output!r}')

    index_start = self._filter_bank_index_start
    index_end = self._filter_bank_index_end
    # convert fft output complex numbers to energy values
    _debug_print_internal(
        f'index start, end [{detail}]: {index_start}, {index_end}')
    # Using type Any to work around Pylance errors
    energy_output: tf.Tensor | Any = energy_op.energy(
        fft_output, index_start, index_end)
    # Energy op does not zero indices outside [index_start,index_end).
    # The following operations to zero portions of the energy op output
    # could be much more efficiently performed inside the energy op C++
    # code.
    # Need to convert to tf.int32 or the TfLite converter will not use
    # the correct ops.
    energy_output = tf.cast(energy_output, tf.int32)
    zeros_head = tf.zeros(index_start, dtype=tf.int32)
    # Using type Any to work around Pylance errors
    number_of_elements: int | Any = energy_output.shape.num_elements()
    zeros_tail = tf.zeros(number_of_elements - index_end, dtype=tf.int32)
    energy_slice = energy_output[index_start:index_end]
    energy_output = tf.concat([zeros_head, energy_slice, zeros_tail], 0)
    energy_output = tf.cast(energy_output, dtype=tf.uint32)
    _debug_print_internal(f'energy output [{detail}]: {energy_output!r}')

    # compress energy output into 40 channels
    filter_output: tf.Tensor = filter_bank_ops.filter_bank(
        energy_output,
        params.sample_rate,
        params.filter_bank_number_of_channels,
        params.filter_bank_lower_band_limit_hz,
        params.filter_bank_upper_band_limit_hz)
    _debug_print_internal(f'filterbank output [{detail}]: {filter_output!r}')

    # scale down filter_output
    filter_scaled_output: tf.Tensor = filter_bank_ops.filter_bank_square_root(
        filter_output, scaling_shift)
    _debug_print_internal(
        f'scaled filterbank output [{detail}]: {filter_scaled_output!r}')

    # noise reduction
    spectral_sub_bits: int = params.filter_bank_spectral_subtraction_bits
    filter_noise_output: tf.Tensor
    filter_noise_estimate: tf.Tensor
    filter_noise_output, filter_noise_estimate = \
        filter_bank_ops.filter_bank_spectral_subtraction(
            filter_scaled_output,
            num_channels=params.filter_bank_number_of_channels,
            smoothing=params.filter_bank_even_smoothing,
            alternate_smoothing=params.filter_bank_odd_smoothing,
            smoothing_bits=params.filter_bank_smoothing_bits,
            min_signal_remaining=params.filter_bank_min_signal_remaining,
            clamping=params.filter_bank_clamping,
            spectral_subtraction_bits=spectral_sub_bits,
        )
    _debug_print_internal(f'noise output [{detail}]: {filter_noise_output!r}')

    # automatic gain control (PCAN)
    correction_bits: int = self._fft_size_log2 - \
        int(params.filter_bank_scaling_bits / 2)
    filter_agc_output: tf.Tensor = pcan_op.pcan(
        filter_noise_output,
        filter_noise_estimate,
        strength=params.pcan_strength,
        offset=params.pcan_offset,
        gain_bits=params.pcan_gain_bits,
        smoothing_bits=params.pcan_smoothing_bits,
        input_correction_bits=correction_bits)
    _debug_print_internal(
        f'AGC Noise output [{detail}]: {filter_agc_output!r}')

    # re-scale features from UINT32 to INT16
    feature_post_scale: int = 1 << params.filter_bank_post_scaling_bits
    feature_pre_scale_shift: int = correction_bits
    feature_rescaled_output: tf.Tensor = filter_bank_ops.filter_bank_log(
        filter_agc_output,
        output_scale=feature_post_scale,
        input_correction_bits=feature_pre_scale_shift)
    _debug_print_internal(
        f'scaled noise output [{detail}]: {feature_rescaled_output!r}')

    # These scaling values are derived from those used in input_data.py in the
    # training pipeline.
    # The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
    # range. In training, these are then arbitrarily divided by 25.6 to get
    # float values in the rough range of 0.0 to 26.0. This scaling is performed
    # for historical reasons, to match up with the output of other feature
    # generators.
    # The process is then further complicated when we quantize the model. This
    # means we have to scale the 0.0 to 26.0 real values to the -128 to 127
    # signed integer numbers.
    # All this means that to get matching values from our integer feature
    # output into the tensor input, we have to perform:
    # input = (((feature / 25.6) / 26.0) * 256) - 128
    # To simplify this and perform it in 32-bit integer math, we rearrange to:
    # input = (feature * 256) / (25.6 * 26.0) - 128
    # constexpr int32_t value_scale = 256;
    # constexpr int32_t value_div =
    #     static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    # int32_t value =
    #     ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
    #     value_div;
    # value -= 128;
    # if (value < -128) {
    #   value = -128;
    # }
    # if (value > 127) {
    #   value = 127;
    # }
    # output[i] = value;

    value_scale = tf.constant(256, dtype=tf.int32)
    value_div = tf.constant(int((25.6 * 26) + 0.5), dtype=tf.int32)
    # Using type Any to work around Pylance errors
    feature_output: tf.Tensor | Any = tf.cast(
        feature_rescaled_output, tf.int32)
    feature_output = (feature_output * value_scale) + int(value_div / 2)
    feature_output = tf.truncatediv(feature_output, value_div)
    feature_output += tf.constant(-128, dtype=tf.int32)
    feature_output = tf.clip_by_value(
        feature_output, clip_value_min=-128, clip_value_max=127)
    feature_output = tf.cast(feature_output, tf.int8)
    _debug_print_internal(f'feature output [{detail}]: {feature_output!r}')

    self._debug_single = True

    return feature_output


@dataclass(kw_only=True)
class FeatureParams:
  """
  Feature generator parameters

  Defaults are configured to work the the micro_speech_quantized.tflite model
  """

  sample_rate: int = 16000
  """audio sample rate"""

  window_size_ms: int = 30
  """input window size in milliseconds"""

  window_stride_ms: int = 20
  """input window stride in milliseconds"""

  window_scaling_bits: int = 12
  """input window shaping: scaling bits"""

  filter_bank_number_of_channels: int = 40
  """filter bank channel count"""

  filter_bank_lower_band_limit_hz: float = 125.0
  """filter bank lower band limit"""

  filter_bank_upper_band_limit_hz: float = 7500.0
  """filter bank upper band limit"""

  filter_bank_scaling_bits: int = \
      filter_bank_ops.FILTER_BANK_WEIGHT_SCALING_BITS
  """filter bank weight scaling bits, updates filter bank constant"""

  filter_bank_alignment: int = 4
  """filter bank alignment, updates filter bank constant"""

  filter_bank_channel_block_size: int = 4
  """filter bank channel block size, updates filter bank constant"""

  filter_bank_post_scaling_bits: int = 6
  """filter bank output log-scaling bits"""

  filter_bank_spectral_subtraction_bits: int = 14
  """filter bank noise reduction spectral subtration bits"""

  filter_bank_smoothing_bits: int = 10
  """filter bank noise reduction smoothing bits"""

  filter_bank_even_smoothing: float = 0.025
  """filter bank noise reduction even smoothing"""

  filter_bank_odd_smoothing: float = 0.06
  """filter bank noise reduction odd smoothing"""

  filter_bank_min_signal_remaining: float = 0.05
  """filter bank noise reduction minimum signal remaining"""

  filter_bank_clamping: bool = False
  """filter bank noise reduction clamping"""

  pcan_strength: float = 0.95
  """PCAN gain control strength"""

  pcan_offset: float = 80.0
  """PCAN gain control offset"""

  pcan_gain_bits: int = 21
  """PCAN gain control bits"""

  pcan_smoothing_bits = 10
  """PCAN gain control smoothing bits"""

  legacy_output_scaling: float = 25.6
  """Final output scaling, legacy from training"""

  def __post_init__(self):
    pass


class AudioPreprocessor:
  """Audio Preprocessor"""

  def __init__(
          self,
          params: FeatureParams = FeatureParams(),
          detail: str = 'unknown'):
    self._detail = detail
    self._params = params
    self._samples_per_window: int = 0
    self._tflm_interpreter = None
    self._feature_generator = None
    self._feature_generator_concrete_function = None
    self._model = None

  def _get_feature_generator(self):
    if self._feature_generator is None:
      self._feature_generator = _GenerateFeature(
          name='GenerateFeature',
          params=self._params,
          detail=self._detail)
    return self._feature_generator

  def _get_concrete_function(self):
    if self._feature_generator_concrete_function is None:
      shape = [1, self._samples_per_window]
      fg = self._get_feature_generator()
      func = tf.function(func=fg.generate_feature_for_frame)
      self._feature_generator_concrete_function = func.get_concrete_function(
          tf.TensorSpec(shape=shape, dtype=tf.int16))  # type: ignore
    return self._feature_generator_concrete_function

  def _get_model(self):
    if self._model is None:
      cf = self._get_concrete_function()
      converter = tf.lite.TFLiteConverter.from_concrete_functions(
          [cf], self._get_feature_generator())
      converter.allow_custom_ops = True
      self._model = converter.convert()
      if _ENABLE_DEBUG.value != 'off':
        tf.lite.experimental.Analyzer.analyze(model_content=self._model)
    return self._model

  def load_samples(self, filename: Path, use_rounding: bool = False):
    file_data = tf.io.read_file(str(filename))
    # Using type Any to work around Pylance errors
    samples: tf.Tensor | Any
    samples, sample_rate = tf.audio.decode_wav(file_data, desired_channels=1)
    sample_rate = int(sample_rate)
    _debug_print(f'Loaded {filename.name}'
                 f' sample-rate={sample_rate}'
                 f' sample-count={len(samples)}')
    assert sample_rate == self._params.sample_rate, 'mismatched sample rate'
    # convert samples to INT16
    # i = (((int) ((x * 32767) + 32768.5f)) - 32768);
    max_value = tf.dtypes.int16.max
    min_value = tf.dtypes.int16.min
    if use_rounding:
      samples = ((samples * max_value) + (-min_value + 0.5)) + min_value
    else:
      samples *= -min_value
    samples = tf.cast(samples, tf.int16)
    samples = tf.reshape(samples, [1, -1])

    self._samples = samples
    self._samples_per_window = self._params.window_size_ms * \
        int(sample_rate / 1000)

    # reset for new graph generation
    self._feature_generator = None
    self._feature_generator_concrete_function = None
    self._model = None
    self._tflm_interpreter = None

  @property
  def samples(self) -> tf.Tensor:
    return self._samples

  @property
  def params(self) -> FeatureParams:
    return self._params

  def generate_feature(self, audio_frame: tf.Tensor) -> tf.Tensor:
    fg = self._get_feature_generator()
    feature = fg.generate_feature_for_frame(audio_frame=audio_frame)
    return feature

  def generate_feature_using_graph(self, audio_frame: tf.Tensor) -> tf.Tensor:
    cf = self._get_concrete_function()
    # Using type Any to work around Pylance errors
    feature: tf.Tensor | Any = cf(audio_frame=audio_frame)
    return feature

  def generate_feature_using_tflm(self, audio_frame: tf.Tensor) -> tf.Tensor:
    if self._tflm_interpreter is None:
      model = self._get_model()
      self._tflm_interpreter = runtime.Interpreter.from_bytes(model)

    self._tflm_interpreter.set_input(audio_frame, 0)
    self._tflm_interpreter.invoke()
    result = self._tflm_interpreter.get_output(0)
    return tf.convert_to_tensor(result)

  def reset_tflm(self):
    if self._tflm_interpreter is not None:
      self._tflm_interpreter.reset()

  def generate_tflite_file(self, type_name):
    model = self._get_model()
    fname = Path('/tmp', 'audio_preprocessor_' + type_name + '.tflite')
    with open(fname, mode='wb') as file_handle:
      file_handle.write(model)


_FeatureGenFunc = Callable[[tf.Tensor], tf.Tensor]


def _compare_test_with_tflm_reset(
        pp: AudioPreprocessor,
        f1: _FeatureGenFunc,
        name: str):
  feature1 = f1(pp.samples)
  for i in range(1, 4):
    pp.reset_tflm()  # reset all internal state
    feature2 = pp.generate_feature_using_tflm(pp.samples)
    tf.debugging.assert_equal(
        feature1, feature2, message=f'{name}: iteration {i}')
  _debug_print(f'{name}: {feature1!r}')


def _compare_test(
        pp: AudioPreprocessor,
        f1: _FeatureGenFunc,
        name: str):
  feature1 = None
  for i in range(1, 50):
    feature1 = f1(pp.samples)
    feature2 = pp.generate_feature_using_tflm(pp.samples)
    tf.debugging.assert_equal(
        feature1, feature2, message=f'{name}: iteration {i}')
  _debug_print(f'{name}: {feature1!r}')


def _main(_):
  prefix_path = resource_loader.get_path_to_datafile('testdata')

  fname = _FILE_TO_TEST.value
  audio_30ms_path = Path(prefix_path, f'{fname}_30ms.wav')

  pp = AudioPreprocessor(detail=fname)
  pp.load_samples(audio_30ms_path)

  #
  # Tests take into account the inability to reset the noise estimation
  # state during TensorFlow eager and graph execution.
  #
  _compare_test_with_tflm_reset(
      pp,
      pp.generate_feature,
      f'{fname}_features_using_func with reset')
  _compare_test(
      pp,
      pp.generate_feature,
      f'{fname}_features_using_func')
  pp.reset_tflm()  # reset all internal state
  _compare_test(
      pp,
      pp.generate_feature_using_graph,
      f'{fname}_features_using_graph')

  print(f'\n[{fname}]: All tests PASS\n')

  pp.generate_tflite_file('quantized')


if __name__ == '__main__':
  app.run(_main)
