# ComfyUI-LeMiCa

**LeMiCa** has been integrated into **ComfyUI** and is fully compatible with ComfyUI native nodes. **ComfyUI-LeMiCa** is easy to use — simply connect the **LeMiCa** node with ComfyUI native nodes for seamless integration.


<p align="center">
  <img src="ComfyUI-LeMiCa\assets\ComfyUI.png" alt="ComfyUI-LeMiCa Workflow" width="800"/>
</p>


### supported models
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)  
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) 


## Installation

```bash
# 1. Go to the ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes/

# 2. Copy ComfyUI-LeMiCa into the custom_nodes directory
# (replace ComfyUI-LeMiCa with the actual path)
cp -r ComfyUI-LeMiCa ./

# 3. Install dependencies
cd ComfyUI-LeMiCa
pip install -r requirements.txt
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
This project is built upon the foundations of [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI), [**ComfyUI-TeaCache**](https://github.com/welltop-cn/ComfyUI-TeaCache) and [**Diffusers**](https://github.com/huggingface/diffusers).  We gratefully acknowledge the efforts of all contributors to these repositories.