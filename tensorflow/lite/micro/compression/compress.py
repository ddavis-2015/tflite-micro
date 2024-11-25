# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import bitarray
import bitarray.util
from collections.abc import ByteString
from dataclasses import dataclass, field
import sys
from typing import Iterable

import absl.app
import absl.flags
import flatbuffers
import numpy as np

from tflite_micro.tensorflow.lite.micro.compression import model_facade
from tflite_micro.tensorflow.lite.micro.compression import spec
from tflite_micro.tensorflow.lite.micro.compression import metadata_py_generated as schema

USAGE = f"""\
Usage: compress.py --input <in.tflite> --spec <spec.yaml> [--output <out.tflite>]

Produce a compressed model from the input model by compressing tensors
according to the instructions in the spec file. The spec file lists the tensors
to compress, the compression methods to use on each tensor, and any parameters
for each compression method.

The spec file is a YAML-format file with a dictionary at the root, containing a
key "tensors" with a list of tensors to compress as its value. E.g.:

---
{spec.EXAMPLE_YAML_SPEC}
---

The only compression method currently implemented is "lut", i.e.,
Look-Up-Table. This method requires the tensor in the input model to have a
small number of unique values, fewer than or equal to 2**index_bitwidth. LUT
compression collects these values into a lookup table, and rewrites the tensor
as bitwidth-wide integer indices into that lookup table. Presumably, the input
model has been trained or preprocessed in a way that the tensor values
are binned into a meaningful, limited set.
"""

TFLITE_METADATA_KEY = "COMPRESSION_METADATA"


class CompressionError(Exception):
  """Raised when compression fails for the reason documented in the message."""

  def __init__(self, message, wrapped_exception=None):
    super().__init__(f"{message}: {str(wrapped_exception)}")
    self.original_exception = wrapped_exception


class _MetadataBuilder:

  def __init__(self):
    self._metadata = schema.MetadataT()
    self._metadata.subgraphs = []

  def compile(self) -> bytearray:
    """Packs the metadata into a binary array and returns it.
    """
    builder = flatbuffers.Builder(1 * 2**10)
    root = self._metadata.Pack(builder)
    builder.Finish(root)
    return builder.Output()

  def subgraph(self, index: int):
    """Return subgraph at index, adding subgraphs if necessary.
    """
    while len(self._metadata.subgraphs) <= index:
      self._add_subgraph()
    return self._metadata.subgraphs[index]

  def add_lut_tensor(self, subgraph_id: int):
    """Add LUT tensor to the given subgraph and return it.
    """
    tensor = schema.LutTensorT()
    self.subgraph(subgraph_id).lutTensors.append(tensor)
    return tensor

  def _add_subgraph(self):
    subgraph = schema.SubgraphT()
    subgraph.lutTensors = []
    self._metadata.subgraphs.append(subgraph)
    return subgraph


@dataclass
class LutCompressedArray:
  compression_axis: int = 0
  lookup_tables: list[np.ndarray] = field(default_factory=list)
  indices: np.ndarray = field(default_factory=lambda: np.array([]))

  @property
  def index_bitwidth(self) -> int:
    """Returns the number of bits required to encode the indices."""
    if self.indices is None:
      raise ValueError

    max_index = np.max(self.indices)
    return int(np.ceil(np.log2(max_index) or 1))


def _lut_compress_array(tensor: np.ndarray, axis: int) -> LutCompressedArray:
  """Compresses using a lookup table per subarray along the given axis.

  Compressing a tensor with a lookup table per subarray along a particular axis
  is analogous to quantizing a tensor with different quantization parameters
  per subarray along a particular axis (dimension).
  """
  compressed = LutCompressedArray()
  compressed.compression_axis = axis

  # Iterate over subarrays along the compression axis
  subarray_indices = []
  for subarray in np.moveaxis(tensor, axis, 0):
    values, indices = np.unique(subarray, return_inverse=True)
    compressed.lookup_tables.append(values)
    indices = indices.reshape(subarray.shape)
    subarray_indices.append(indices)

  # Reconstruct a tensor of indices from the subarrays
  stacked = np.stack(subarray_indices, axis=0)
  compressed.indices = np.moveaxis(stacked, 0, axis)

  return compressed


def _assert_lut_only(compression):
  if len(compression) != 1:
    raise CompressionError("Each tensor must have exactly one compression")
  if not isinstance(compression[0], spec.LookUpTableCompression):
    raise CompressionError('Only "lut" compression may be specified')


def _identify_compression_axis(tensor: model_facade._Tensor) -> int:
  """Finds the axis along which to compress.

  Use the quantization axis, else the NWHC channel dimension. If necessary,
  an user-specified override could be added to the compression spec schema.
  """
  if tensor.quantization is not None:
    axis = tensor.quantization.quantizedDimension
  else:
    axis = tensor.array.ndim - 1

  return axis


def _check_bitwidth(compressed: int, specified: int, spec: spec.Tensor):
  """Applies business logic regarding specified bitwidth.

  It is an error if the bitwidth required to compress a tensor exceeds the
  specified bitwith, and a warning if the tensor can be compressed in less than
  the specified bitwidth. The latter is allowed, and is not an error, to permit
  testing with larger bitwidths without re-binning a model.
  """
  if compressed > specified:
    raise CompressionError(
        f"index_bitwidth too small: {compressed} bits needed to "
        f"enumerate unique values in tensor specified in {spec}")
  elif compressed < specified:
    print(
        f"warning: index_bitwidth too large: only {compressed} "
        f"bits needed to enumerate unique values in tensor specified in {spec}",
        file=sys.stderr)


def _pack_indices(indices: np.ndarray, bitwidth: int) -> bytes:
  """Packs indices into a bytearray using bitwidth-sized fields.
  """
  endianness = "big"
  bits = bitarray.bitarray(endian=endianness)
  for i in indices.ravel():
    bits.extend(
        bitarray.util.int2ba(int(i), length=bitwidth, endian=endianness))
  return bits.tobytes()


def _pack_lookup_tables(tables: list[np.ndarray], table_len: int) -> bytearray:
  """Packs the value tables of a LutCompressedArray.

  Pack the value tables of a LutCompressedArray into a bytes object in the
  format writable to a value_table buffer in the .tflite flatbuffer. The
  tables, one per subarray, are concatinated.
  """
  buffer = bytearray()
  for t in tables:
    padding_needed = table_len - len(t)
    padded = np.pad(t, (0, padding_needed), mode='constant', constant_values=0)
    buffer.extend(padded.tobytes())

  return buffer


def compress(model_in: ByteString, specs: Iterable[spec.Tensor]) -> bytearray:
  model = model_facade.read(model_in)
  metadata = _MetadataBuilder()

  for spec in specs:
    try:
      tensor = model.subgraphs[spec.subgraph].tensors[spec.tensor]
      _assert_lut_only(spec.compression)
      axis = _identify_compression_axis(tensor)
      compressed = _lut_compress_array(tensor.array, axis)
      spec_bitwidth = spec.compression[0].index_bitwidth
      _check_bitwidth(compressed.index_bitwidth, spec_bitwidth, spec)

      # overwrite tensor data with indices
      tensor.buffer.data = _pack_indices(compressed.indices, spec_bitwidth)

      # write value buffer
      value_buffer = model.add_buffer()
      value_buffer.data = _pack_lookup_tables(compressed.lookup_tables,
                                              2**spec_bitwidth)
      # add compression metadata for tensor
      lut_tensor = metadata.add_lut_tensor(subgraph_id=tensor.subgraph.index)
      lut_tensor.tensor = tensor.index
      lut_tensor.valueBuffer = value_buffer.index
      lut_tensor.indexBitwidth = spec_bitwidth

    except Exception as e:
      raise CompressionError(f"error compressing {spec}") from e

  # add compression metadata to model
  model.add_metadata(TFLITE_METADATA_KEY, metadata.compile())

  return model.compile()


FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("input", None, None)
absl.flags.DEFINE_string("spec", None, None)
absl.flags.DEFINE_string("output", None, None)


def _fail_w_usage() -> int:
  absl.app.usage()
  return 1


def main(argv):
  if len(argv) > 1:
    return _fail_w_usage()

  in_path = FLAGS.input
  if in_path is None:
    return _fail_w_usage()
  else:
    with open(in_path, "rb") as in_file:
      in_model = in_file.read()

  spec_path = FLAGS.spec
  if spec_path is None:
    return _fail_w_usage()
  else:
    with open(spec_path, "rb") as spec_file:
      specs = spec.parse_yaml(spec_file.read())

  out_path = FLAGS.output
  if out_path is None:
    out_path = in_path.split(".tflite")[0] + ".compressed.tflite"

  compressed = compress(in_model, specs)

  with open(out_path, "wb") as out_file:
    out_file.write(compressed)

  return 0


if __name__ == "__main__":
  sys.modules['__main__'].__doc__ = USAGE
  absl.app.run(main)
