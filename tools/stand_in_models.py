"""Build the stand-in models that let the neural path run without any real weights.

Three stages of the stack are neural and all three are optional in the product:

    parser   models/face_parser.onnx      semantic masks (occluders preserved)
    swapper  models/inswapper_128.onnx    the swap itself          [non-commercial licence]
    restorer models/gfpgan_1.4.onnx       detail recovery

None of them exist in this repository, and the swap weights cannot legally be shipped in one.
The consequence has been that the neural path — the parser's mask, the compositor's blend, the
boost's phase interleave, the restorer's blend — could only ever be read, never run: every test
around it substitutes a fake in place of the model session.

This writes small ONNX graphs with the **exact shapes and names the adapters require**, built
from hand-written convolution kernels rather than trained weights. They are not faces and not
demonstrations of quality; they are a way to execute the plumbing on a CPU-only host, catch a
shape or layout mistake before a licensed artefact is bought, and prove that a host is capable.

The shapes are the contract, so they are taken from the adapters themselves rather than from
memory:

* `app/compositor.py` reads `get_inputs()[0].shape[-1]` for its size, feeds NCHW RGB in
  [-1, 1], and expects 19 logits whose class layout is the CelebAMask-HQ/BiSeNet one.
* `insightface.model_zoo.inswapper.INSwapper` requires exactly one output, reads
  `graph.initializer[-1]` as the 512x512 embedding map, and runs with
  `{inputs[0]: crop, inputs[1]: latent}` — so input order is part of the contract, not a
  detail.
* `app/enhance.py` feeds NCHW BGR in [0, 1] and blends the result at a visibility weight.

`--directory` defaults to `models/standin/`, deliberately *not* `models/`, so a stand-in is
never mistaken for a licensed artefact by a person or by `tools/fetch_models.py --check`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

#: The 19-class CelebAMask-HQ layout the compositor's class sets are written against.
SKIN_CLASS = 1
LIP_CLASSES = (12, 13)
EYES_CLASS = 4
GLASSES_CLASS = 6
HAIR_CLASS = 17
NECK_CLASS = 15
CLASSES = 19

#: `inswapper_128` is fixed to 128px crops and a 512-wide identity embedding.
ALIGN = 128
EMBEDDING = 512

PARSER_SIZE = 512
#: The label template is written at this resolution and upsampled inside the graph. At 512 the
#: initialiser alone is 19 MB of float32 — a fixture that costs more to build than the code it
#: exercises. The compositor only reads the model's declared input size, so the template's own
#: resolution is free to be small.
PARSER_TEMPLATE = 64
RESTORER_SIZE = 512


def _square(value: float, name_prefix: str = "") -> np.ndarray:
    """A 3x3 kernel that keeps the pixel it is centred on."""
    kernel = np.zeros((3, 3), np.float32)
    kernel[1, 1] = value
    return kernel


def _identity_conv(weight: float = 1.0) -> np.ndarray:
    """Depthwise identity: (3, 1, 3, 3), the same trick `static/lab/swapper-standin.onnx` uses."""
    return np.stack([_square(weight)] * 3)[:, None]


def _sharpen_conv() -> np.ndarray:
    """Identity plus a little unsharp: enough that the restorer's blend is observable."""
    kernel = np.array([[0.0, -0.08, 0.0],
                       [-0.08, 1.32, -0.08],
                       [0.0, -0.08, 0.0]], np.float32)
    return np.stack([kernel] * 3)[:, None]


def _save(nodes: list, inputs: list, outputs: list, initializers: list, path: Path) -> Path:
    graph = helper.make_graph(nodes, path.stem, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def swapper(path: Path) -> Path:
    """A 128px swap stand-in: the crop back, tinted by the source embedding.

    Takes the crop through an identity convolution — preserving high frequencies, which is the
    property the boost's phase interleave is supposed to exploit — then blends in a colour
    derived from the source latent so the output genuinely depends on both inputs. A stand-in
    that ignored the source would pass every shape check and prove nothing.
    """
    target = helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, 3, ALIGN, ALIGN])
    embedding = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, EMBEDDING])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, ALIGN, ALIGN])

    initializers = [
        numpy_helper.from_array(_identity_conv(0.98), "crop_weight"),
        # 512 -> 3: three numbers per embedding, one per channel. `eye(512, 3)` is the
        # (512, 3) shape MatMul needs; written the other way round it is (3, 512) and the
        # graph fails shape inference before the model can be loaded.
        numpy_helper.from_array((np.eye(EMBEDDING, 3, dtype=np.float32) * 0.5), "tint_weight"),
        numpy_helper.from_array(np.array([0.85], np.float32), "crop_share"),
        numpy_helper.from_array(np.array([0.15], np.float32), "tint_share"),
        numpy_helper.from_array(np.array([1, 3, 1, 1], np.int64), "tint_shape"),
        # LAST in the file, and it must really be last: `INSwapper.__init__` loads the model
        # with `onnx.load` and takes `graph.initializer[-1]` as the 512x512 embedding map. Any
        # initializer declared after it is read as the emap instead, and a 4-element array
        # where a 512x512 one belongs fails later with a shape complaint that says nothing
        # about the cause. It is also applied to the latent *here* rather than only in numpy,
        # so ONNX Runtime keeps it: an initializer no node uses is stripped from the session.
        numpy_helper.from_array(np.eye(EMBEDDING, dtype=np.float32), "emap"),
    ]

    nodes = [
        helper.make_node("Conv", ["target", "crop_weight"], ["cropped"], pads=[1, 1, 1, 1],
                         group=3),
        helper.make_node("MatMul", ["source", "emap"], ["projected"]),
        helper.make_node("MatMul", ["projected", "tint_weight"], ["tint"]),
        helper.make_node("Reshape", ["tint", "tint_shape"], ["tint_map"]),
        helper.make_node("Mul", ["cropped", "crop_share"], ["cropped_scaled"]),
        helper.make_node("Mul", ["tint_map", "tint_share"], ["tint_scaled"]),
        helper.make_node("Add", ["cropped_scaled", "tint_scaled"], ["output"]),
    ]
    return _save(nodes, [target, embedding], [output], initializers, path)


def parser(path: Path) -> Path:
    """A 19-class label stand-in with a plausible face layout.

    The logits are a fixed spatial template — skin and lips inside an ellipse, hair across the
    top, glasses over the eyes, ears at the sides — plus a term that depends on the input, so
    the adapter's own preprocessing (resize, [-1, 1] normalisation, argmax) is genuinely in the
    path. The point is that the compositor then has an occluder *to preserve*, which a constant
    label map could not test.
    """
    size = PARSER_TEMPLATE
    template = np.zeros((1, CLASSES, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy, rx, ry = size / 2, size * 0.62, size * 0.30, size * 0.40
    face = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

    template[0, SKIN_CLASS][face] = 6.0
    template[0, LIP_CLASSES[0]][face & (yy > cy + ry * 0.55)] = 7.0
    template[0, LIP_CLASSES[1]][face & (yy > cy + ry * 0.62)] = 7.0
    template[0, EYES_CLASS][face & (np.abs(yy - (cy - ry * 0.15)) < 6)] = 5.0
    template[0, HAIR_CLASS][yy < size * 0.22] = 8.0
    template[0, NECK_CLASS][yy > size * 0.88] = 7.0
    template[0, GLASSES_CLASS][face & (np.abs(yy - (cy - ry * 0.15)) < 14) & (xx < cx + 90)] = 9.0

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                             [1, 3, PARSER_SIZE, PARSER_SIZE])
    output_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                              [1, CLASSES, PARSER_SIZE, PARSER_SIZE])
    initializers = [
        numpy_helper.from_array(template, "template"),
        numpy_helper.from_array(np.array([], np.float32), "roi"),
        numpy_helper.from_array(np.array([1, 1, PARSER_SIZE / size, PARSER_SIZE / size],
                                         np.float32), "scales"),
        numpy_helper.from_array(np.array([0.0], np.float32), "silence"),
    ]
    nodes = [
        # ReduceMean(axes=[1]) collapses (1,3,H,W) to (1,1,H,W) so the template can be
        # broadcast onto it — channel 3 versus channel 19 will not broadcast on its own.
        helper.make_node("ReduceMean", ["input"], ["ink"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["ink", "silence"], ["quiet"]),
        helper.make_node("Resize", ["template", "roi", "scales"], ["expanded"],
                         mode="nearest"),
        helper.make_node("Add", ["expanded", "quiet"], ["logits"]),
    ]
    return _save(nodes, [input_vi], [output_vi], initializers, path)


def restorer(path: Path) -> Path:
    """A detail stand-in: sharpened, clipped to [0, 1], same size in and out."""
    size = RESTORER_SIZE
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, size, size])
    initializers = [
        numpy_helper.from_array(_sharpen_conv(), "sharp_weight"),
        numpy_helper.from_array(np.array([0.0], np.float32), "clip_low"),
        numpy_helper.from_array(np.array([1.0], np.float32), "clip_high"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "sharp_weight"], ["sharp"], pads=[1, 1, 1, 1],
                         group=3),
        helper.make_node("Clip", ["sharp", "clip_low", "clip_high"], ["output"]),
    ]
    return _save(nodes, [input_vi], [output_vi], initializers, path)


#: filename -> builder. The names are the ones `app/config.py` expects, so pointing the
#: settings at this directory needs no renaming.
BUILDERS = {
    "face_parser.onnx": parser,
    "inswapper_128.onnx": swapper,
    "gfpgan_1.4.onnx": restorer,
}

def environment() -> dict[str, str]:
    """With `models/standin` as the swap directory, these env vars make the app load them.

    The backend name is imported rather than written out: `insightface` instead of `inswapper`
    is the kind of mistake that does not raise, it silently runs the diagnostic engine. The
    import happens *here*, when the dict is asked for — not at module import. Read at import
    time it pulled in `app.config` and therefore pydantic-settings, so generating stand-ins, a
    job that needs nothing but numpy and onnx, could not run in a bare environment: on Kaggle
    it died on `ModuleNotFoundError: pydantic_settings` before writing a single file.
    """
    from app.engines.factory import INSWAPPER_BACKEND

    return {
        "STUDIO_MODEL_PATH": "models/standin/inswapper_128.onnx",
        "STUDIO_PARSER_MODEL_PATH": "models/standin/face_parser.onnx",
        "STUDIO_RESTORATION_MODEL_PATH": "models/standin/gfpgan_1.4.onnx",
        "STUDIO_BACKEND": INSWAPPER_BACKEND,
    }


def build(directory: Path) -> dict[str, Path]:
    return {name: build_one(name, directory / name) for name in BUILDERS}


def build_one(name: str, path: Path) -> Path:
    return BUILDERS[name](path)


def main() -> int:
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument("--directory", type=Path, default=Path("models/standin"),
                           help="where to write them (default: models/standin)")
    options = arguments.parse_args()
    for name, path in build(options.directory).items():
        print(f"wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
    print("\nthese are stand-ins: correct shapes, no trained weights, no quality claim.")
    print("to run the stack against them:")
    for key, value in environment().items():
        print(f"  {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
