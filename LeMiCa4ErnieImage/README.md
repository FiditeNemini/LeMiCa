# LeMiCa4ErnieImage

[LeMiCa](https://github.com/UnicomAI/LeMiCa) now supports accelerated inference for **ERNIE-Image** and provides three optional acceleration paths based on the balance between quality and speed.

#### ERNIE-Image

| Method | ERNIE-Image | LeMiCa-slow (B=25) | LeMiCa-medium (B=17) | LeMiCa-fast (B=10) |
|:------:|:-----------:|:------------------:|:--------------------:|:------------------:|
| **Latency** | 32.168 s | 16.471 s | 11.432 s | 7.043 s |
| **T2I** | <img width="160" alt="ERNIE-Image" src="https://github.com/user-attachments/assets/c01d9ef0-df8a-4c7c-bb61-b69d54cfaf9b" /> | <img width="160" alt="LeMiCa-slow" src="https://github.com/user-attachments/assets/25ef5d70-aae9-4664-8e95-59ad5848cb6b" /> | <img width="160" alt="LeMiCa-medium" src="https://github.com/user-attachments/assets/17cf4dfb-4d46-4b87-a8ca-7990064b9252" /> | <img width="160" alt="LeMiCa-fast" src="https://github.com/user-attachments/assets/5790d160-66fe-46a7-a8c4-14c66c6f7888" /> |

## Inference Latency

#### Comparisons on a Single H800

| ERNIE-Image | LeMiCa-slow (B=25) | LeMiCa-medium (B=17) | LeMiCa-fast (B=10) |
|:-----------:|:------------------:|:--------------------:|:------------------:|
| 32.168 s | 16.471 s | 11.432 s | 7.043 s |

## Files

- `inference_ernieimage.py`: LeMiCa-enabled ErnieImage inference entry.

## Installation

Use the same environment as ErnieImage / diffusers:

```bash
pip install -U torch transformers diffusers
```

## Usage

Baseline (no cache):

```bash
python inference_ernieimage.py --model-path baidu/ERNIE-Image
```

LeMiCa acceleration:

```bash
python inference_ernieimage.py --model-path baidu/ERNIE-Image --cache slow
python inference_ernieimage.py --model-path baidu/ERNIE-Image --cache medium
python inference_ernieimage.py --model-path baidu/ERNIE-Image --cache fast
```

Equivalent explicit B-steps:

```bash
python inference_ernieimage.py --model-path baidu/ERNIE-Image --cache 25
python inference_ernieimage.py --model-path baidu/ERNIE-Image --cache 17
python inference_ernieimage.py --model-path baidu/ERNIE-Image --cache 10
```

Common options:

```bash
python inference_ernieimage.py \
  --model-path baidu/ERNIE-Image \
  --cache medium \
  --prompt "A cinematic shot of a robot cat in rain." \
  --height 1024 --width 1024 \
  --num-inference-steps 50 \
  --guidance-scale 4.0 \
  --seed 42
```

## Path Configuration Note

Current cache paths are initialized from QwenImage-style settings:
- `slow -> B=25`
- `medium -> B=17`
- `fast -> B=10`

When ErnieImage-specific optimal paths are available, only the path dictionary in `inference_ernieimage.py` needs to be updated.

## 📖 Citation

If you find **LeMiCa** useful in your research or applications, please consider giving us a star and citing it by the following BibTeX entry:

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

We would like to thank the contributors to [ERNIE-Image](https://github.com/PaddlePaddle/ERNIE), [TeaCache](https://github.com/ali-vilab/TeaCache), and [Diffusers](https://github.com/huggingface/diffusers).
