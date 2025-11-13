<!-- ## **LeMiCa4Wan2.1** -->
# LeMiCa4Wan2.1

[LeMiCa](https://github.com/UnicomAI/LeMiCa) extends its acceleration framework to [Wan2.1](https://github.com/Wan-Video/Wan2.1), providing **training-free inference acceleration** for both T2V and I2V pipelines. It achieves **up to 2× faster generation** while preserving visual quality, by introducing lexicographic minimax path caching for efficient diffusion reuse.


---

## 🎥 Visualization Results and Inference Latency 

LeMiCa introduces an adjustable parameter **`--lemica_budget (B)`**,  
which controls the number of cached denoising paths used during inference.  
By providing different budgets, users can flexibly trade off **generation quality** and **speed**.

Below are four representative Wan2.1 model examples accelerated by **LeMiCa**.  



---

### 🧩 Text-to-Video (T2V)


| Model | Wan2.1 (latency min) | LeMiCa (B=25) | LeMiCa (B=20) | LeMiCa (B=17) |
|:------:|:--------------------:|:--------------:|:--------------:|:--------------:|
| **T2V 1.3B 480p** | $1.76$ | $0.95$ ($\mathbf{1.85\text{x}}$) | $0.79$ ($\mathbf{2.23\text{x}}$) | $0.68$ ($\mathbf{2.59\text{x}}$) |
| **T2V 14B 720p**  | $31.48$ | $15.95$ ($\mathbf{1.97\text{x}}$) | $12.85$ ($\mathbf{2.45\text{x}}$) | $10.98$ ($\mathbf{2.87\text{x}}$) |

#### T2V 1.3B
https://github.com/user-attachments/assets/c465c0f4-99a1-4fe4-b61e-44ff21bb9ee3

#### T2V 14B
https://github.com/user-attachments/assets/bcdaa756-b020-4e37-9f37-2dfd1fefb88a

---

### 🧩 Image-to-Video (I2V)

| Model | Wan2.1 (latency min) | LeMiCa (B=25) | LeMiCa (B=20) | LeMiCa (B=17) | LeMiCa (B=14) |
|:------:|:--------------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| **I2V 14B 480p** | $7.78$ | $5.01$ ($\mathbf{1.55\text{x}}$) | $4.44$ ($\mathbf{1.75\text{x}}$) | $3.62$ ($\mathbf{2.15\text{x}}$) | $3.24$ ($\mathbf{2.40\text{x}}$) |
<!-- | **T2V 14B 720p**  | $31.48$ | $15.95$ ($\mathbf{1.97\text{x}}$) | $12.85$ ($\mathbf{2.45\text{x}}$) | $10.98$ ($\mathbf{2.87\text{x}}$) | $10.98$ ($\mathbf{2.87\text{x}}$) | -->

####  I2V 480P
https://github.com/user-attachments/assets/9c6ab4be-ce6d-482e-b2fa-e5ee673a66b9

<!-- ####  I2V 720P
https://github.com/user-attachments/assets/bcdaa756-b020-4e37-9f37-2dfd1fefb88a -->




## ⚙️ Usage

Follow [Wan2.1](https://github.com/Wan-Video/Wan2.1) for installation and model preparation,  
then copy **`inference_wan.py`** from this repository into the Wan2.1 root directory.


---

### Example Commands


### Note
The `--lemica_budget` parameter supports multiple values (e.g., 25, 20, 17, 14),  
where a smaller budget provides **higher acceleration** at the cost of **slightly reduced visual quality**.  
If not specified (default: `None`), the model reverts to the **original full denoising process** without any acceleration.


```bash
# T2V 1.3B 480p
python inference_wan.py \
  --task t2v-1.3B --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 42 --offload_model True --t5_cpu \
  --lemica_budget 25


# T2V 14B 720p
python inference_wan.py \
  --task t2v-14B --size 1280*720 \
  --ckpt_dir ./Wan2.1-T2V-14B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 42 --offload_model True --t5_cpu \
  --lemica_budget 25


# I2V 14B 480p
python inference_wan.py \
  --task i2v-14B --size 832*480 \
  --ckpt_dir ./Wan2.1-I2V-14B-480P \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 42 --offload_model True --t5_cpu \
  --lemica_budget 25


# I2V 14B 720p
python inference_wan.py \
  --task i2v-14B --size 1280*720 \
  --ckpt_dir ./Wan2.1-I2V-14B-720P \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 42 --offload_model True --t5_cpu --frame_num 61 \
  --lemica_budget 25


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

We would like to thank the contributors to the  [Wan2.1](https://github.com/Wan-Video/Wan2.1),  [TeaCache](https://github.com/ali-vilab/TeaCache) and [Diffusers](https://github.com/huggingface/diffusers).