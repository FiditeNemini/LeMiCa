# ⚡ FLUX.2 Inference Acceleration

[**FLUX2**](https://github.com/black-forest-labs/flux2) is a high-performance **text-to-image** and **image-to-image** diffusion framework built by *Black Forest Labs*, optimized for multi-GPU & large-batch inference. It supports advanced caching & remote encoding strategies for maximum throughput 🚀.

---

## 📊 Inference Latency

**Example latency (single H100/H800 @ 1024×1024):**

| Method              | Flux.2 (CPU-offload) | Flux.2         | LeMiCa-slow    | LeMiCa-medium | LeMiCa-fast   |
|:-------------------:|:--------------------:|:--------------:|:--------------:|:-------------:|:-------------:|
| **Latency (sec)**   | 101.2                | 32.70          | 13.41          | 10.20         | 6.99          |
| **T2I** | <img width="120" alt="Flux2 CPU-offload" src="https://github.com/user-attachments/assets/76fda91e-8819-4914-87e4-8a832135da0f" /> | <img width="120" alt="Flux2" src="https://github.com/user-attachments/assets/a3f320e3-9d36-4618-9953-f714646e6bf7" /> | <img width="120" alt="LeMiCa-slow" src="https://github.com/user-attachments/assets/b28fdd2b-8178-4ba7-bf23-3da66f555593" /> | <img width="120" alt="LeMiCa-medium" src="https://github.com/user-attachments/assets/72b4361d-8afe-4c94-9654-77697e3c1444" /> | <img width="120" alt="LeMiCa-fast" src="https://github.com/user-attachments/assets/56ea6af3-e1a5-4134-890b-24f5666081e9" /> |


> 💡 Numbers above are example measurements; actual latency may vary depending on resolution, batch size, and hardware configuration.

---

## 🛠️ Installation & Usage


Please refer to the original [**FLUX.2**](https://github.com/black-forest-labs/flux2) project for base installation instructions.

### 📡 Remote Text Encoder (H100/H800 Recommended)
For heavy text encoding workloads, you can offload the text encoder to a separate dedicated GPU (e.g., H100) via a simple FastAPI service.

```bash
def remote_text_encoder(prompts, device):
    """
    Calls the recently deployed FastAPI service and returns prompt_embeds (torch.Tensor).
    prompts: str or List[str]
    """
    TEXT_ENCODER_URL = "http://127.0.0.1:8006/predict"
    
    resp = requests.post(
        TEXT_ENCODER_URL,
        json={"prompt": prompts},
        timeout=600,
    )
    resp.raise_for_status()

    # Use torch.load for deserialization, same as in the official example
    prompt_embeds = torch.load(io.BytesIO(resp.content))

    # Move to the device used for current inference
    return prompt_embeds.to(device)    

```

### Usage 

```bash
# vanilla FLUX.2 (no caching / acceleration)
python inference_flux2.py

# LeMiCa acceleration modes
python inference_flux2.py --cache slow
python inference_flux2.py --cache medium
python inference_flux2.py --cache fast

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