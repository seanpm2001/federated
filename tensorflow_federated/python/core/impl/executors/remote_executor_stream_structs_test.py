# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types


class RemoteExecutorStreamStructsTest(parameterized.TestCase):

  def test_large_struct_identity0(self):
    small_tensor_shape = (100000000, 1)
    large_struct = structure.Struct(
        [(None, tf.zeros(shape=small_tensor_shape, dtype=tf.float32))] * 6)

    @tensorflow_computation.tf_computation(
        computation_types.StructType(
            [(None,
              computation_types.TensorType(
                  shape=small_tensor_shape, dtype=tf.float32))] * 6))
    def identity(s):
      with tf.compat.v1.control_dependencies(
          [tf.print(t) for t in structure.flatten(s)]):
        return structure.map_structure(tf.identity, s)

    identity(large_struct)

  def test_large_struct_identity1(self):
    small_tensor_shape = (100000, 1000)
    large_struct = structure.Struct([
        ('a', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('b', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('c', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('d', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('e', tf.zeros(shape=small_tensor_shape, dtype=tf.float32)),
        ('f', tf.zeros(shape=small_tensor_shape, dtype=tf.float32))
    ])

    @tensorflow_computation.tf_computation(
        computation_types.StructType([('a',
                                       computation_types.TensorType(
                                           shape=small_tensor_shape,
                                           dtype=tf.float32)),
                                      ('b',
                                       computation_types.TensorType(
                                           shape=small_tensor_shape,
                                           dtype=tf.float32)),
                                      ('c',
                                       computation_types.TensorType(
                                           shape=small_tensor_shape,
                                           dtype=tf.float32)),
                                      ('d',
                                       computation_types.TensorType(
                                           shape=small_tensor_shape,
                                           dtype=tf.float32)),
                                      ('e',
                                       computation_types.TensorType(
                                           shape=small_tensor_shape,
                                           dtype=tf.float32)),
                                      ('f',
                                       computation_types.TensorType(
                                           shape=small_tensor_shape,
                                           dtype=tf.float32))]))
    def identity(s):
      with tf.compat.v1.control_dependencies(
          [tf.print(t) for t in structure.flatten(s)]):
        return structure.map_structure(tf.identity, s)

    identity(large_struct)


if __name__ == '__main__':
  execution_contexts.set_localhost_cpp_execution_context(stream_structs=True)
  absltest.main()
