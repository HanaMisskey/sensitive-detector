# Model Notices

Source model: `KanariKanaru/nsfw-image-detection-384`
Source revision: `dcbee2f0570c16c3212bc6b81bb8911194b9fa62`
Source license: Apache-2.0
Converted artifact: `model.onnx`
Converted SHA256: `e9350e576608afe4b57a089ffeb0ebafa1389cdcea4882dd61df28c45f1c24d2`

This package contains converted model weights only. It does not contain training data.

Validation:
- logits max absolute diff: 7.152557373046875e-07
- scores max absolute diff: 5.960464477539063e-08
- top label match: 1

Source README excerpt:

---
tags:
- image-classification
- timm
library_name: timm
license: apache-2.0
---
# Model card for nsfw-image-detection-384

__NOTE: Like all models, this one can make mistakes. NSFW content can be subjective and contextual, this model is intended to help identify this content, use at your own risk.__

`Marqo/nsfw-image-detection-384` is a lightweight image classification model designed to identify NSFW images. The model is approximately 18–20x smaller than other open-source models and achieves a superior accuracy of 98.56% on our dataset. This model uses 384x384 pixel images for the input with 16x16 pixel patches.
