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
"""compare feature generator output"""

import argparse
from collections.abc import Generator
from collections import OrderedDict
from typing import TextIO
import re
import functools
import operator

ParsedFile = OrderedDict[str, list[int]]

get_next_line: Generator[str] = None


def get_next_line_generator(file: TextIO) -> Generator[str]:
  for line in file:
    yield line


def parse_tensor_data() -> list[int]:
  values_list: list[int] = []
  pattern = r'(-?\d+)[,\]]\s*'
  pattern_re = re.compile(pattern)

  for line in get_next_line:
    values: list[str] = pattern_re.findall(line)
    values_list += values
    if ']' in line:
      break

  values_list = list(map(int, values_list))
  return values_list


def parse_dump_data() -> list[int]:
  values_list: list[int] = []
  pattern = r'(-?\d+),\s*'
  pattern_re = re.compile(pattern)

  for line in get_next_line:
    values: list[str] = pattern_re.findall(line)
    if len(values) == 0:
      break
    values_list += values

  values_list = list(map(int, values_list))
  return values_list


def parse_flat_size(line: str) -> int:
  # print(f'line: {line}')
  shape_pattern = r'shape=\((?:\d+,?\s?)+\)'
  match = re.search(shape_pattern, line)
  # print(f'match: {match!r}')
  digit_pattern = r'\d+'
  result: list[str] = re.findall(digit_pattern, match.group(0))
  # print(f'shape dims: {result!r}')
  flat_size = functools.reduce(lambda x, y: int(x) * int(y), result, 1)
  return flat_size


def parse_file(file: TextIO) -> ParsedFile:
  global get_next_line
  get_next_line = get_next_line_generator(file)
  parsed_map: ParsedFile = OrderedDict()
  tensor_pattern = r'(?:(\w+\s*\w*) output|(feature)s) \[(\w+)\]: <tf.Tensor:'
  dump_pattern = r'### BEGIN (\w+\s*\w*) Data: (\w+)'
  tensor_re = re.compile(tensor_pattern)
  dump_re = re.compile(dump_pattern)

  for line in get_next_line:
    # print(line)
    match_tensor = tensor_re.match(line)
    # print(f'match_tensor: {match_tensor!r}')
    match_dump = dump_re.match(line)
    # print(f'match_dump: {match_dump!r}')
    key = None
    value = []

    if match_tensor is not None:
      # print(f'groups:{match_tensor.groups()!r}')
      if match_tensor.group(1) is not None:
        key = match_tensor.expand(r'\1 [\3]').lower()
      else:
        key = match_tensor.expand(r'\2 [\3]').lower()
      value = parse_tensor_data()
      flat_size = parse_flat_size(line)
      if flat_size != len(value):
        print(f'data size ({len(value)}) != flat size ({flat_size})')
        continue
    elif match_dump is not None:
      # print(f'groups:{match_dump.groups()!r}')
      key = match_dump.expand(r'\1 [\2]').lower()
      value = parse_dump_data()
    else:
      continue

    parsed_map[key] = value

  return parsed_map


def find_missing_keys(file1: ParsedFile, file2: ParsedFile):
  for key in file1.keys():
    if key not in file2.keys():
      print(f'key {{{key}}} missing from second file')
  for key in file2.keys():
    if key not in file1.keys():
      print(f'key {{{key}}} missing from first file')


def compare_parsed_files(file1: ParsedFile, file2: ParsedFile):
  for key in file1.keys():
    if key not in file2.keys():
      continue
    if file1[key] == file2[key]:
      print(f'key {{{key}}} matches\n')
      continue
    diff = list(map(operator.sub, file1[key], file2[key]))
    print(f'key {{{key}}} differs:\n{diff}\n')


def main():
  parser = argparse.ArgumentParser(
      description='Compare micro_features_dump output with'
      ' generate_features output')
  parser.add_argument('file1', metavar='file1.txt', type=argparse.FileType())
  parser.add_argument('file2', metavar='file2.txt', type=argparse.FileType())
  args = parser.parse_args()

  parsed_file1 = parse_file(args.file1)
  parsed_file2 = parse_file(args.file2)
  # print(f'{parsed_file1!r}')
  # print(f'{parsed_file2!r}')
  find_missing_keys(parsed_file1, parsed_file2)
  compare_parsed_files(parsed_file1, parsed_file2)


if __name__ == '__main__':
  main()
