<!-- ## **LeMiCa4Wan2.1** -->
# LeMiCa4HunyuanVideo-1.5

[LeMiCa](https://github.com/UnicomAI/LeMiCa) extends its acceleration framework to [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), providing **training-free inference acceleration** for both T2V and I2V pipelines. It achieves **up to 2.8× faster generation** while preserving visual quality, by introducing lexicographic minimax path caching for efficient diffusion reuse.


---

## 🎥 Visualization Results and Inference Latency 

LeMiCa introduces an adjustable parameter **`--lemica_step (B)`**, which controls the number of cached denoising paths used during inference. By providing different budgets, users can flexibly trade off **generation quality** and **speed**. Below are two representative HunyuanVideo-1.5 model examples accelerated by **LeMiCa**. Since our primary focus is acceleration on the **denoising step count**, we did not include other optimization techniques such as **SageAttention**, **sparse attention**, and similar methods in the comparison.


---

### 🧩 Text-to-Video (T2V) H100x4


| Model | HunyuanVideo-1.5 (latency min) | LeMiCa (B=25) | LeMiCa (B=20) | LeMiCa (B=15) |
|:-----:|:------------------------------:|:-------------:|:-------------:|:-------------:|
| **T2V 720p** | **8.98** | **4.84** (**1.85x**) | **4.03** (**2.23x**) | **3.14** (**2.86x**) |


#### T2V 720P
https://github.com/user-attachments/assets/ebed2e0f-87f4-408e-98e3-93bd29bbc99f



---

### 🧩 Image-to-Video (I2V) H100x4

| Model | HunyuanVideo-1.5 (latency min) | LeMiCa (B=25) | LeMiCa (B=20) | LeMiCa (B=15) | LeMiCa (B=10) |
|:-----:|:------------------------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **I2V 720p** | **9.10** | **4.92** (**1.85x**) | **4.04** (**2.25x**) | **3.17** (**2.87x**) | **2.35** (**3.88x**) |

####  I2V 720P
https://github.com/user-attachments/assets/d1a83d45-579f-4174-9477-ba0b9aebb322



## ⚙️ Usage

Follow [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) for installation and model preparation, then replace the following scripts:

```bash
# Copy these files from this repo into the HunyuanVideo-1.5 root:

# 1) Replace infer_state.py
cp infer_state.py ./hyvideo/commons/infer_state.py

# 2) Replace hunyuan_video_pipeline.py
cp hunyuan_video_pipeline.py ./hyvideo/pipelines/hunyuan_video_pipeline.py

# 3) Replace generate.py
cp generate.py ./hyvideo/pipelines/generate.py
```

---

### Example Commands


### Note
The `--lemica_step` parameter supports multiple values (e.g., 25, 20, 15, 10), where a smaller budget provides **higher acceleration** at the cost of **slightly reduced visual quality**.  


```bash

export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A girl holding a paper with words "Hello, world!"'

IMAGE_PATH=/path/to/image.png # Optional, none or <image path> to enable i2v mode
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
OUTPUT_PATH=./outputs/output.mp4
MODEL_PATH=./ckpts # Path to pretrained model

# Configuration for faster inference
N_INFERENCE_GPU=4 # Parallel inference GPU count
CFG_DISTILLED=false # Inference with CFG distilled model, 2x speedup
SAGE_ATTN=false # Inference with SageAttention
SPARSE_ATTN=false # Inference with sparse attention (only 720p models are equipped with sparse attention). Please ensure flex-block-attn is installed
OVERLAP_GROUP_OFFLOADING=true # Only valid when group offloading is enabled, significantly increases CPU memory usage but speeds up inference
ENABLE_CACHE=true # Enable feature cache during inference. Significantly speeds up inference.
CACHE_TYPE=lemica # Support: deepcache, teacache, taylorcache, lemica
LEMICA_STEP=25  # lemica step (25,20,15,10)

ENABLE_STEP_DISTILL=false # Enable step distilled model for 480p I2V, recommended 8 or 12 steps, up to 6x speedup
# Configuration for better quality
REWRITE=false # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution


torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --rewrite $REWRITE \
  --cfg_distilled $CFG_DISTILLED \
  --enable_step_distill $ENABLE_STEP_DISTILL \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --sr $ENABLE_SR --save_pre_sr_video \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --lemica_step $LEMICA_STEP

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

We would like to thank the contributors to the  [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5),  [TeaCache](https://github.com/ali-vilab/TeaCache) and [Diffusers](https://github.com/huggingface/diffusers).