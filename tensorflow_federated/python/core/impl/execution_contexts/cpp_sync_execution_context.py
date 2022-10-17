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
"""A context for execution based on an embedded executor instance."""

from typing import Any, Callable, Optional

from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.execution_contexts import cpp_async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import execution_context
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.executors import executor_factory as executor_factory_lib


class SyncSerializeAndExecuteCPPContext(execution_context.SyncExecutionContext):
  """A synchronous execution context delegating to CPP Executor bindings."""

  def __init__(
      self,
      factory: executor_factory_lib.ExecutorFactory,
      compiler_fn: Optional[Callable[[computation_base.Computation],
                                     Any]] = None,
      *,
      cardinality_inference_fn: cardinalities_utils
      .CardinalityInferenceFnType = cardinalities_utils.infer_cardinalities):
    self._async_execution_context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
        factory, compiler_fn, cardinality_inference_fn=cardinality_inference_fn)
    self._async_runner = async_utils.AsyncThreadRunner()

  @property
  def executor_factory(self) -> executor_factory_lib.ExecutorFactory:
    return self._async_execution_context.executor_factory

  def invoke(self, comp, arg):
    return self._async_runner.run_coro_and_return_result(
        self._async_execution_context.invoke(comp, arg))
