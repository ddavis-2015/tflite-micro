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

from pathlib import Path
import tensorflow as tf
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import resource_loader
from tflite_micro.python.tflite_micro.signal.ops import window_op
from tflite_micro.python.tflite_micro.signal.ops import fft_ops
from tflite_micro.python.tflite_micro.signal.ops import energy_op
from tflite_micro.python.tflite_micro.signal.ops import filter_bank_ops

_PREFIX_PATH = resource_loader.get_path_to_datafile('')


def load_samples(filename: Path) -> tuple[tf.Tensor, int]:
  file_data = tf.io.read_file(str(filename))
  samples: tf.Tensor
  samples, sample_rate = tf.audio.decode_wav(file_data, desired_channels=1)
  sample_rate = int(sample_rate)
  print(f'Loaded {filename.name}'
        f' sample-rate={sample_rate}'
        f' sample-count={len(samples)}')
  # convert samples to INT16
  # i = (((int) ((x * 32767) + 32768.5f)) - 32768);
  max_value = np.iinfo(np.int16).max
  min_value = np.iinfo(np.int16).min
  samples = ((samples * max_value) + (-min_value + 0.5)) + min_value
  samples = tf.cast(samples, tf.int16)
  samples = tf.reshape(samples, [1, -1])

  return samples, sample_rate


def generate_features_for_frame(audio_frame: tf.Tensor,
                                sample_rate: int) -> tf.Tensor:
  # apply window to audio frame
  window_shift = 12
  window_sample_count = 30 * sample_rate / 1000
  weights: np.ndarray = window_op.hann_window_weights(
      window_sample_count, window_shift)
  print(f'audio frame shape: {audio_frame.get_shape()}')
  print(f'weights shape: {weights.shape}')
  window_output: tf.Tensor = window_op.window(
      audio_frame, weights, window_shift)
  # print(f'window output: {window_output!r}')

  # pre-scale window output
  window_output = tf.reshape(window_output, [-1])
  window_scaled_output, scaling_shift = fft_ops.fft_auto_scale(window_output)
  print(f'scaling shift: {scaling_shift!r}')
  # print(f'window_scaled_output: {window_scaled_output!r}')

  # compute FFT on scaled window output
  fft_size, _ = fft_ops.get_pow2_fft_length(len(window_scaled_output))
  print(f'fft size: {fft_size}')
  fft_output = fft_ops.rfft(window_scaled_output, fft_size)

  # convert fft output complex numbers to energy values
  number_of_channels = 40
  lower_band_limit_hz = 125.0
  upper_band_limit_hz = 7500.0
  index_start, index_end = filter_bank_ops.calc_start_end_indices(
      fft_size, sample_rate, number_of_channels,
      lower_band_limit_hz, upper_band_limit_hz)
  print(f'index start, end: {index_start}, {index_end}')
  energy_output: tf.Tensor = energy_op.energy(
      fft_output, index_start, index_end)

  # compress energy output into 40 channels
  filter_output: tf.Tensor = filter_bank_ops.filter_bank(
      energy_output, sample_rate, number_of_channels,
      lower_band_limit_hz, upper_band_limit_hz)

  # scale down filter_output
  filter_scaled_output: tf.Tensor = filter_bank_ops.filter_bank_square_root(
      filter_output, scaling_shift)

  # noise reduction

  # automatic gain control

  # re-scale features from UINT32 to UINT16
  # int correction_bits =
  #   MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  #   (10 - 1 - (12 / 2))
  feature_post_scale_shift = 6
  feature_pre_scale_shift = 3
  feature_rescaled_output: tf.Tensor = filter_bank_ops.filter_bank_log(
      filter_scaled_output, feature_post_scale_shift, feature_pre_scale_shift)

  return feature_rescaled_output


def main(_):
  no_30ms_path = Path(_PREFIX_PATH, 'testdata/no_30ms.wav')
  yes_30ms_path = Path(_PREFIX_PATH, 'testdata/yes_30ms.wav')

  print(f'Paths:\n{no_30ms_path}\n{yes_30ms_path}')
  no_samples, no_sample_rate = load_samples(no_30ms_path)
  no_features = generate_features_for_frame(no_samples, no_sample_rate)
  yes_samples, yes_sample_rate = load_samples(yes_30ms_path)
  yes_features = generate_features_for_frame(yes_samples, yes_sample_rate)

  print(f'no features: {no_features!r}')
  print(f'yes features: {yes_features!r}')


if __name__ == '__main__':
  app.run(main)