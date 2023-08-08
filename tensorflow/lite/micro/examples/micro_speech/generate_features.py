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
  window_shift = 12
  window_sample_count = 30 * sample_rate / 1000
  weights: np.ndarray = window_op.hann_window_weights(
      window_sample_count, window_shift)
  print(f'audio frame shape: {audio_frame.get_shape()}')
  print(f'weights shape: {weights.shape}')
  window_output: tf.Tensor = window_op.window(
      audio_frame, weights, window_shift)

  print(f'window output: {window_output!r}')
  return window_output[0, 0:40]


def main(_):
  no_30ms_path = Path(_PREFIX_PATH, 'testdata/no_30ms.wav')
  yes_30ms_path = Path(_PREFIX_PATH, 'testdata/yes_30ms.wav')

  print(f'Paths:\n{no_30ms_path}\n{yes_30ms_path}')
  no_samples, no_sample_rate = load_samples(no_30ms_path)
  no_features = generate_features_for_frame(no_samples, no_sample_rate)
  yes_samples, yes_sample_rate = load_samples(yes_30ms_path)
  yes_features = generate_features_for_frame(yes_samples, yes_sample_rate)

  print(f'no features: {no_features}')
  print(f'yes features: {yes_features}')


if __name__ == '__main__':
  app.run(main)
