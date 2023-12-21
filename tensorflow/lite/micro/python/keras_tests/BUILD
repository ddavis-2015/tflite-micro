# Description:
#   TensorFlow Lite microcontroller example.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

py_binary(
    name = "audio_preprocessor",
    srcs = ["audio_preprocessor.py"],
    data = [
        ":samples_30ms",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//python/tflite_micro:runtime",
        "//python/tflite_micro/signal:ops",
        "//python/tflite_micro/signal/utils:util",
    ],
)

py_test(
    name = "conv_tests",
    srcs = ["conv_tests.py"],
    main = "conv_tests.py",
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//python/tflite_micro:runtime",
    ],
)
