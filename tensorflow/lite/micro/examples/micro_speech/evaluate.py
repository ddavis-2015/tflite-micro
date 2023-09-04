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
# =============================================================================
"""
LSTM model evaluation for MNIST recognition

Run:
bazel build tensorflow/lite/micro/examples/mnist_lstm:evaluate
bazel-bin/tensorflow/lite/micro/examples/mnist_lstm/evaluate
--model_path=".tflite file path" --img_path="MNIST image path"
"""


from absl import app
from absl import flags
from absl import logging
import numpy as np
from pathlib import Path

from tflite_micro.python.tflite_micro import runtime
from tensorflow.python.platform import resource_loader
import tensorflow as tf
import audio_preprocessor


_SAMPLE_PATH = flags.DEFINE_string(
    name='sample_path',
    default=None,
    help='path for the audio sample to be predicted.',
    required=True
)

_SAMPLE_RATE = 16000
_FEATURES_SHAPE = (49, 40)
_WINDOW_SIZE_MS = 30
_WINDOW_STRIDE_MS = 20


def quantize_input_data(data, input_details):
  """quantize the input data using scale and zero point

  Args:
      data (np.array in float): input data for the interpreter
      input_details : output of get_input_details from the tflm interpreter.
  """
  # Get input quantization parameters
  data_type = input_details["dtype"]
  input_quantization_parameters = input_details["quantization_parameters"]
  input_scale, input_zero_point = input_quantization_parameters["scales"][
      0], input_quantization_parameters["zero_points"][0]
  # quantize the input data
  data = data / input_scale + input_zero_point
  return data.astype(data_type)


def dequantize_output_data(data: np.ndarray,
                           output_details: dict) -> np.ndarray:
  """Dequantize the model output

  Args:
      data: integer data to be dequantized
      output_details: TFLM interpreter model output details

  Returns:
      np.ndarray: dequantized data as float32 dtype
  """
  output_quantization_parameters = output_details["quantization_parameters"]
  output_scale = output_quantization_parameters["scales"][0]
  output_zero_point = output_quantization_parameters["zero_points"][0]
  # Caveat: tflm_output_quant need to be converted to float to avoid integer
  # overflow during dequantization
  # e.g., (tflm_output_quant -output_zero_point) and
  # (tflm_output_quant + (-output_zero_point))
  # can produce different results (int8 calculation)
  return output_scale * (data.astype("float") - output_zero_point)


def tflm_predict(
        interpreter: runtime.Interpreter,
        data: np.ndarray) -> np.ndarray:
  """
  Predict using the tflm interpreter

  Args:
      tflm_interpreter (Interpreter): TFLM interpreter
      data: data to be predicted

  Returns:
      np.ndarray: predicted results from the model using TFLM interpreter
  """
  interpreter.set_input(data, 0)
  interpreter.invoke()
  return interpreter.get_output(0)


def predict(interpreter: runtime.Interpreter,
            features: np.ndarray) -> np.ndarray:
  """
  Use TFLM interpreter to predict wake-word from audio sample features

  Args:
      interpreter: TFLM python interpreter instance
      features: data to be predicted

  Returns:
      np.ndarray: predicted probability (softmax) for each model category
  """

  # input_details = interpreter.get_input_details(0)
  # # Quantize the input if the model is quantized
  # if input_details["dtype"] != np.float32:
  #   data = quantize_input_data(data, input_details)
  interpreter.set_input(features, 0)
  interpreter.invoke()
  tflm_output = interpreter.get_output(0)

  output_details = interpreter.get_output_details(0)
  if output_details["dtype"] == np.float32:
    return tflm_output[0].astype("float")
  # Dequantize the output for quantized model
  return dequantize_output_data(tflm_output[0], output_details)


def generate_features(
        audio_pp: audio_preprocessor.AudioPreprocessor) -> np.ndarray:
  """
  Generate audio sample features

  Args:
      audio_pp: AudioPreprocessor instance

  Returns:
      np.ndarray: generated audio sample features with shape _FEATURES_SHAPE
  """
  features = np.zeros(_FEATURES_SHAPE, dtype=np.int8)
  start_index = 0
  window_size = int(_WINDOW_SIZE_MS * _SAMPLE_RATE / 1000)
  window_stride = int(_WINDOW_STRIDE_MS * _SAMPLE_RATE / 1000)
  samples = audio_pp.samples[0]
  frame_number = 0
  end_index = start_index + window_size

  # clear noise estimates
  audio_pp.reset_tflm()

  while end_index <= len(samples):
    frame_tensor: tf.Tensor = tf.convert_to_tensor(
        samples[start_index:end_index])
    frame_tensor = tf.reshape(frame_tensor, [1, -1])
    feature_tensor = audio_pp.generate_feature_using_tflm(frame_tensor)
    features[frame_number] = feature_tensor.numpy()
    start_index += window_stride
    end_index += window_stride
    frame_number += 1
  return features


def predict_sample(interpreter: runtime.Interpreter, sample_path: Path):
  """
  Use TFLM interpreter to predict a audio sample

  Args:
      interpreter: TFLM python interpreter instance
      audio_path: path for the audio sample that will be predicted

  Returns:
      np.ndarray: predicted probability (softmax) for each model category
  """
  audio_pp = audio_preprocessor.AudioPreprocessor()
  audio_pp.load_samples(sample_path)
  assert audio_pp.sample_rate == _SAMPLE_RATE, \
      f'Audio sample rate must be {_SAMPLE_RATE} per second'
  features = generate_features(audio_pp)
  flattened_features = features.flatten().reshape([1, -1])
  return features, predict(interpreter, flattened_features)


def main(_):
  sample_path = Path(_SAMPLE_PATH.value)
  assert sample_path.exists() and sample_path.is_file(), \
      'Audio sample file does not exist. Please check the path.'
  model_prefix_path = resource_loader.get_path_to_datafile('models')
  model_path = Path(model_prefix_path, 'micro_speech_quantized.tflite')

  tflm_interpreter = runtime.Interpreter.from_file(model_path)
  features, category_probabilities = predict_sample(
      tflm_interpreter, sample_path)
  frame_number = 0
  for feature in features:
    logging.info('Frame #%d: %s', frame_number, str(feature))
    frame_number += 1
  predicted_category = np.argmax(category_probabilities)
  category_names = ['silence', 'unknown', 'yes', 'no']
  logging.info('Model predicts the audio sample as <%s> with probability %.2f',
               category_names[predicted_category],
               category_probabilities[predicted_category])


if __name__ == '__main__':
  app.run(main)
