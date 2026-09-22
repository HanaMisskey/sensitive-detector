# NSFW Image Detection 384 — ONNX

This directory bundles `model.onnx` from
[KanariKanaru/nsfw-image-detection-384-onnx](https://huggingface.co/KanariKanaru/nsfw-image-detection-384-onnx),
revision `8edc47eedf74b30fd379673bc202fe3b754b1538`.
The original model is [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384).

The model is Apache-2.0 licensed. See `LICENSE-APACHE-2.0.txt` and `MODEL_NOTICES.md`.
`manifest.json` is copied from the upstream package. `SHA256SUMS` contains the
model checksum verified against that manifest.

## Interface

- Input: `input`, float32 `[1, 3, 384, 384]` (NCHW), RGB.
- Normalize each channel with `(pixel / 255 - 0.5) / 0.5`.
- Output: `logits`, float32 `[1, 2]`; index 0 is `nsfw`, index 1 is `safe`.
- Apply numerically stable softmax to obtain probabilities.
- The caller must rotate, flatten transparency, and resize directly to 384×384
  using Catmull–Rom (bicubic) before sending a PNG. The service does not resize.

## Reproduce the download

From this directory:

```sh
curl -L --fail https://huggingface.co/KanariKanaru/nsfw-image-detection-384-onnx/resolve/8edc47eedf74b30fd379673bc202fe3b754b1538/model.onnx -o model.onnx
shasum -a 256 -c SHA256SUMS
```

This replaces the previous InceptionV3 model. Callers must migrate from 299×299
and the five nsfwjs classes to 384×384 and `nsfw`/`safe`. Thresholds need to be
selected using representative images; old class scores are not interchangeable.
