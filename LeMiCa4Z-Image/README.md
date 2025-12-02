 # LeMiCa4Z-Image

[LeMiCa](https://github.com/UnicomAI/LeMiCa) already supports accelerated inference for [Z-Image](https://github.com/Tongyi-MAI/Z-Image) and provides three optional acceleration paths that balance image quality and speed.


## 📊 Inference Latency
#### Comparisons on a Single H800

| Z-Image | LeMiCa-slow | LeMiCa-medium | LeMiCa-fast |
|:-------:|:-----------:|:-------------:|:-----------:|
| 2.55 s  | 2.19 s      | 1.94 s        | 1.78 s      |
| <img width="120" alt="Z-Image" src="https://github.com/user-attachments/assets/e7aa76a9-2ffd-4cfc-8c9d-2240f357850b" /> | <img width="120" alt="LeMiCa-slow" src="https://github.com/user-attachments/assets/e7ff50b9-44bb-48ff-86f9-14dacc1b5144" /> | <img width="120" alt="LeMiCa-medium" src="https://github.com/user-attachments/assets/786ad801-ac92-4467-86a6-661b5e7dca53" /> | <img width="120" alt="LeMiCa-fast" src="https://github.com/user-attachments/assets/722d79b1-69fb-4683-914f-e92533394393" /> |

> Note: The above numbers are example latency measurements for a single H800 GPU with a 1024×1024 resolution. Actual performance may vary depending on hardware and configuration.

## 🛠️ Installation & Usage

Please refer to the original [Z-Image]([Z-Image](https://github.com/Tongyi-MAI/Z-Image) ) project for base installation instructions.


```bash

# vanilla Z-Image (no caching / acceleration)
python inference_zimage.py

# LeMiCa acceleration modes
python inference_zimage.py --cache slow
python inference_zimage.py --cache medium
python inference_zimage.py --cache fast

# use an explicit numeric cache step
python inference_zimage.py --cache 8
```



## 📖 Citation
If you find **LeMiCa** useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:

```bibtex
@inproceedings{gao2025lemica,
  title     = {LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation},
  author    = {Huanlin Gao and Ping Chen and Fuyuan Shi and Chao Tan and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
  journal   = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.00090}
}
```

## Acknowledgements

We would like to thank the contributors to the  [Z-Image](https://github.com/Tongyi-MAI/Z-Image) and [Diffusers](https://github.com/huggingface/diffusers).