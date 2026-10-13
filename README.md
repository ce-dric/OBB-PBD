# OBB-PBD

**OBB-PBD: Extending Parallel Box Decoding to Oriented Bounding Boxes for Referring Expression Grounding in Aerial Imagery**

Chang Woo Song, Yeong Jun Koh (Chungnam National University)

*Remote Sensing* (MDPI), accepted, 2026.

## Release status

The code, the training configuration and the evaluation protocol will be released in this repository when the
article is published. The release will also include the source diff of the five fixed upstream defects against the
pinned upstream revision, and the package inventory of the training container.

The checkpoints of all nine trained runs (OBB-PBD and the two matched controls, three seeds each: LoRA adapters plus
the fully trained `mlp1` projector) will be released on the Hugging Face Hub at
[huggingface.co/ce-dric/OBB-PBD](https://huggingface.co/ce-dric/OBB-PBD).

## License

The code and checkpoints are derivative works of
[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) and will be distributed under the NVIDIA
License (non-commercial research and evaluation use only). The language model of LocateAnything-3B is based on
Qwen2.5-3B-Instruct, licensed under the Qwen RESEARCH LICENSE AGREEMENT. The RefGeo (AVVG) data are under the S-Lab
License 1.0.

## Citation

The citation will be added when the article is published.
