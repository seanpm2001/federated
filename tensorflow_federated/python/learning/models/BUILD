load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":models_packages",
    "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
    "//tensorflow_federated/python/learning/metrics:metrics_packages",
    "//tensorflow_federated/python/learning/programs:programs_packages",
    "//tensorflow_federated/python/learning/reconstruction:reconstruction_packages",
    "//tensorflow_federated/python/learning/templates:templates_packages",

    # TODO(b/151441025): remove these edges after moving the symbols into more appropriate locations.
    "//tensorflow_federated/python/learning:learning_packages",
    "//tensorflow_federated/python/learning/framework:framework_packages",
])

package_group(
    name = "models_packages",
    packages = ["//tensorflow_federated/python/learning/models/..."],
)

licenses(["notice"])

py_library(
    name = "model_weights",
    srcs = ["model_weights.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/learning:model",
    ],
)

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":functional",
        ":model_weights",
        ":serialization",
    ],
)

py_library(
    name = "functional",
    srcs = ["functional.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning/metrics:finalizer",
        "//tensorflow_federated/python/learning/metrics:keras_utils",
        "//tensorflow_federated/python/tensorflow_libs:variable_utils",
    ],
)

py_test(
    name = "functional_test",
    srcs = ["functional_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":functional",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/tensorflow_libs:variable_utils",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":functional",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_serialization",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/learning:model",
    ],
)

py_test(
    name = "model_weights_test",
    srcs = ["model_weights_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_weights",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:model",
    ],
)

py_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":functional",
        ":serialization",
        ":test_models",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_examples",
    ],
)

py_library(
    name = "test_models",
    testonly = True,
    srcs = ["test_models.py"],
    deps = [
        ":functional",
        "//tensorflow_federated/python/learning:model",
    ],
)
