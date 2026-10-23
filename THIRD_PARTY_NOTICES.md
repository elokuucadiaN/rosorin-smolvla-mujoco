# Third-party material and provenance

Project integration code is provided under the Apache License 2.0, except material identified below. No ownership or new license is claimed over third-party assets.

- **LeRobot / SmolVLA**: Hugging Face, Apache-2.0. Runtime dependency pinned to [commit 0b067df](https://github.com/huggingface/lerobot/tree/0b067df57d21d3a02d6c511f1609172fa39ac29b). It is installed as a dependency rather than copied into this repository.
- **SmolVLM2-500M-Video-Instruct**: Hugging Face, [Apache-2.0 model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct). Its configuration/tokenizer and retained model layers are used by the released fine-tuned policy. The task checkpoint uses simulated demonstrations; upstream pretraining retains its original provenance.
- **ROSOrin Pro robot assets**: meshes under `assets/meshes/` and the factory URDF derive from the ROSOrin Pro vendor materials supplied with this project. The original `rosorin_description` package declares `TODO: License declaration`, rather than an explicit open-source license. These assets retain their original rights and are **not relicensed by this project's Apache-2.0 license**. Source hashes and conversion parameters are recorded under `assets/factory_baseline/`. No claim is made that vendor geometry is newly authored here.
- **MuJoCo, NumPy, SciPy, PyTorch, Transformers, Pillow, safetensors**: separately installed dependencies, governed by their respective licenses.

This repository contains no real-robot reference video, training-server credentials, private SSH host keys, training data collection, or optimizer/checkpoint history. The demonstration video shows the simulation only.
