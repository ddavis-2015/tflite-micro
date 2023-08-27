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

from typing import Callable
from pathlib import Path
import tensorflow as tf
from absl import app
from absl import flags
import numpy as np
from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import window_op
from tflite_micro.python.tflite_micro.signal.ops import fft_ops
from tflite_micro.python.tflite_micro.signal.ops import energy_op
from tflite_micro.python.tflite_micro.signal.ops import filter_bank_ops
from tflite_micro.python.tflite_micro import runtime


_ENABLE_DEBUG = flags.DEFINE_bool(
    'enable_debug',
    False,
    'Enable debug output',
)
_FILE_TO_TEST = flags.DEFINE_enum('file_to_test',
                                  'no',
                                  ['no', 'yes'],
                                  'File to test')


def _debug_print(*args):
  if _ENABLE_DEBUG.value:
    print(*args)


class _GenerateFeature(tf.Module):
  """Generate feature tensor from audio window samples"""

  def __init__(self, name=None, sample_rate=0, detail='unknown'):
    super().__init__(name=name)
    self._sample_rate = sample_rate
    self._detail = detail
    self._window_shift = 12
    window_sample_count = 30 * sample_rate / 1000
    self._hann_window_weights = tf.constant(window_op.hann_window_weights(
        window_sample_count, self._window_shift))

  def generate_feature_for_frame(self, audio_frame: tf.Tensor) -> tf.Tensor:
    _debug_print('*** generate_feature_for_frame ***')
    sample_rate = self._sample_rate
    detail = self._detail

    _debug_print(f'audio frame output [{detail}]: {audio_frame!r}')

    # apply window to audio frame
    window_shift = self._window_shift
    weights = self._hann_window_weights
    _debug_print(f'window weights output [{detail}]: {weights!r}')
    window_output: tf.Tensor = window_op.window(
        audio_frame, weights, window_shift)
    _debug_print(f'window output [{detail}]: {window_output!r}')
    # return window_output

    # pre-scale window output
    window_output = tf.reshape(window_output, [-1])
    window_scaled_output, scaling_shift = fft_ops.fft_auto_scale(window_output)
    _debug_print(f'scaling shift [{detail}]: {scaling_shift!r}')
    # _debug_print(f'window scaled output: {window_scaled_output!r}')

    # compute FFT on scaled window output
    fft_size, _ = fft_ops.get_pow2_fft_length(len(window_scaled_output))
    _debug_print(f'fft size [{detail}]: {fft_size}')
    fft_output = fft_ops.rfft(window_scaled_output, fft_size)
    _debug_print(f'fft output [{detail}]: {fft_output!r}')

    # convert fft output complex numbers to energy values
    number_of_channels = 40
    lower_band_limit_hz = 125.0
    upper_band_limit_hz = 7500.0
    index_start, index_end = filter_bank_ops.calc_start_end_indices(
        fft_size, sample_rate, number_of_channels,
        lower_band_limit_hz, upper_band_limit_hz)
    _debug_print(f'index start, end [{detail}]: {index_start}, {index_end}')
    energy_output: tf.Tensor = energy_op.energy(
        fft_output, index_start, index_end)
    _debug_print(f'energy output [{detail}]: {energy_output!r}')

    # compress energy output into 40 channels
    filter_output: tf.Tensor = filter_bank_ops.filter_bank(
        energy_output, sample_rate, number_of_channels,
        lower_band_limit_hz, upper_band_limit_hz)
    _debug_print(f'filterbank output [{detail}]: {filter_output!r}')

    # scale down filter_output
    filter_scaled_output: tf.Tensor = filter_bank_ops.filter_bank_square_root(
        filter_output, scaling_shift)
    _debug_print(
        f'scaled filterbank output [{detail}]: {filter_scaled_output!r}')

    # noise reduction
    # config.noise_reduction.smoothing_bits = 10;
    # config.noise_reduction.even_smoothing = 0.025;
    # config.noise_reduction.odd_smoothing = 0.06;
    # config.noise_reduction.min_signal_remaining = 0.05;
    noise_reduction_bits = 14
    smoothing_bits = 10
    even_smoothing = 0.025
    odd_smoothing = 0.06
    min_signal_remaining = 0.05
    filter_noise_output, _ = filter_bank_ops.filter_bank_spectral_subtraction(
        filter_scaled_output,
        num_channels=number_of_channels,
        smoothing=even_smoothing,
        alternate_smoothing=odd_smoothing,
        smoothing_bits=smoothing_bits,
        min_signal_remaining=min_signal_remaining,
        clamping=False,
        spectral_subtraction_bits=noise_reduction_bits,
    )
    _debug_print(f'noise output [{detail}]: {filter_noise_output!r}')

    # automatic gain control (TBD)

    # re-scale features from UINT32 to UINT16
    # int correction_bits =
    #   MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
    #   (10 - 1 - (12 / 2))
    feature_post_scale_shift = 6
    feature_post_scale = 1 << feature_post_scale_shift
    feature_pre_scale_shift = 3
    feature_rescaled_output: tf.Tensor = filter_bank_ops.filter_bank_log(
        filter_noise_output,
        output_scale=feature_post_scale,
        input_correction_bits=feature_pre_scale_shift)
    _debug_print(
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
    # constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
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

    value_scale = 256
    value_div = int((25.6 * 26) + 0.5)
    feature_output = ((tf.cast(feature_rescaled_output, tf.int32)
                      * value_scale) + (value_div // 2)) // value_div
    feature_output -= 128
    feature_output = tf.clip_by_value(
        feature_output, clip_value_min=-128, clip_value_max=127)
    feature_output = tf.cast(feature_output, tf.int8)
    _debug_print(f'feature output [{detail}]: {feature_output!r}')

    return feature_output


class AudioPreprocessor():
  """Audio Preprocessor"""

  def __init__(self, detail='unknown'):
    self._detail = detail
    self._samples = None
    self._sample_rate = 0
    self._tflm_interpreter = None
    self._feature_generator = None
    self._feature_generator_concrete_function = None
    self._samples_per_window_ms = 30
    self._samples_per_window = 0

  def _get_feature_generator(self):
    if self._feature_generator is None:
      self._feature_generator = _GenerateFeature(
          name='GenerateFeature',
          sample_rate=self._sample_rate,
          detail=self._detail)
    return self._feature_generator

  def _get_concrete_function(self):
    if self._feature_generator_concrete_function is None:
      shape = [1, self._samples_per_window]
      fg = self._get_feature_generator()
      func = tf.function(func=fg.generate_feature_for_frame)
      self._feature_generator_concrete_function = func.get_concrete_function(
          tf.TensorSpec(shape=shape, dtype=tf.int16))
    return self._feature_generator_concrete_function

  def load_samples(self, filename: Path):
    file_data = tf.io.read_file(str(filename))
    samples: tf.Tensor
    samples, sample_rate = tf.audio.decode_wav(file_data, desired_channels=1)
    sample_rate = int(sample_rate)
    _debug_print(f'Loaded {filename.name}'
                 f' sample-rate={sample_rate}'
                 f' sample-count={len(samples)}')
    # convert samples to INT16
    # i = (((int) ((x * 32767) + 32768.5f)) - 32768);
    max_value = np.iinfo(np.int16).max
    min_value = np.iinfo(np.int16).min
    samples = ((samples * max_value) + (-min_value + 0.5)) + min_value
    samples = tf.cast(samples, tf.int16)
    samples = tf.reshape(samples, [1, -1])

    self._samples = samples
    self._sample_rate = sample_rate
    self._samples_per_window = self._samples_per_window_ms * \
        (sample_rate // 1000)
    self._feature_generator = None
    self._feature_generator_concrete_function = None

  @property
  def samples(self):
    return self._samples

  @property
  def sample_rate(self):
    return self._sample_rate

  def generate_feature(self, audio_frame: tf.Tensor) -> tf.Tensor:
    fg = self._get_feature_generator()
    feature = fg.generate_feature_for_frame(audio_frame=audio_frame)
    return feature

  def generate_feature_using_graph(self, audio_frame: tf.Tensor) -> tf.Tensor:
    cf = self._get_concrete_function()
    feature = cf(audio_frame=audio_frame)
    return feature

  def generate_feature_using_tflm(self, audio_frame: tf.Tensor) -> tf.Tensor:
    if self._tflm_interpreter is None:
      cf = self._get_concrete_function()
      converter = tf.lite.TFLiteConverter.from_concrete_functions(
          [cf], self._get_feature_generator())
      converter.allow_custom_ops = True
      model = converter.convert()
      if _ENABLE_DEBUG.value:
        tf.lite.experimental.Analyzer.analyze(model_content=model)
      self._tflm_interpreter = runtime.Interpreter.from_bytes(model)

    self._tflm_interpreter.set_input(audio_frame, 0)
    self._tflm_interpreter.invoke()
    result = self._tflm_interpreter.get_output(0)
    return tf.convert_to_tensor(result)

  def reset_tflm(self):
    if self._tflm_interpreter is not None:
      self._tflm_interpreter.reset()


_FeatureGenFunc = Callable[[tf.Tensor], tf.Tensor]


def _compare_test_with_tflm_reset(
        pp: AudioPreprocessor,
        f1: _FeatureGenFunc,
        name='unknown'):
  feature1 = f1(pp.samples)
  for i in range(1, 4):
    pp.reset_tflm()
    feature2 = pp.generate_feature_using_tflm(pp.samples)
    tf.debugging.assert_equal(
        feature1, feature2, message=f'{name}: iteration {i}')
  _debug_print(f'{name}: {feature1!r}')


def _compare_test(
        pp: AudioPreprocessor,
        f1: _FeatureGenFunc,
        name='unknown'):
  for i in range(1, 50):
    feature1 = f1(pp.samples)
    feature2 = pp.generate_feature_using_tflm(pp.samples)
    tf.debugging.assert_equal(
        feature1, feature2, message=f'{name}: iteration {i}')
  _debug_print(f'{name}: {feature1!r}')


def _main(_):
  prefix_path = resource_loader.get_path_to_datafile('')

  fname = _FILE_TO_TEST.value
  audio_30ms_path = Path(prefix_path, f'testdata/{fname}_30ms.wav')

  pp = AudioPreprocessor(detail=fname)
  pp.load_samples(audio_30ms_path)
  _compare_test_with_tflm_reset(
      pp,
      pp.generate_feature,
      f'{fname}_features_using_func with reset')
  _compare_test(
      pp,
      pp.generate_feature,
      f'{fname}_features_using_func')
  pp.reset_tflm()
  _compare_test(
      pp,
      pp.generate_feature_using_graph,
      f'{fname}_features_using_graph')

  print(f'\nAll [{fname}] tests PASS\n')


if __name__ == '__main__':
  app.run(_main)
