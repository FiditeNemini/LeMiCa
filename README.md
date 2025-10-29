# [NeurIPS 2025 Spotlight] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block">
    <a href="https://github.com/joelulu" target="_blank">Huanlin Gao</a><sup>1,2</sup><sup>*</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&user=gpNOW2UAAAAJ" target="_blank">Ping Chen</a><sup>1,2</sup><sup>*</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/stone002" target="_blank">Fuyuan Shi</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/tanchaow" target="_blank">Chao Tan</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=en&user=L4OXOs0AAAAJ" target="_blank">Zhaoxiang Liu</a><sup>1,2</sup>
  </span>
  <br>
  <span class="author-block">
    <a href="https://github.com/FangGet" target="_blank">Fang Zhao</a><sup>1,2</sup><sup>†</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CFUQLCAAAAAJ&hl=en" target="_blank">Kai Wang</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com.hk/citations?user=kCC2oKwAAAAJ&hl=zh-CN&oi=ao" target="_blank">Shiguo Lian</a><sup>1,2</sup>
  </span>
</div>

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block"><sup>1</sup>Data Science & Artificial Intelligence Research Institute, China Unicom,&nbsp;</span>
  <span class="author-block"><sup>2</sup>Unicom Data Intelligence, China Unicom</span>
</div>

<div class="is-size-5 publication-authors" align="center">
  (* Equal contribution. † Corresponding author.)
</div>

<h5 align="center">

<a href="https://unicomai.github.io/LeMiCa/" target="_blank">
  <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Page">
</a>
<a href="https://github.com/UnicomAI/LeMiCa" target="_blank">
  <img src="https://img.shields.io/badge/Code-GitHub-black.svg?logo=github" alt="Code">
</a>
<a href="./assets/LeMiCa_NeurIPS2025.pdf" target="_blank">
  <img src="https://img.shields.io/badge/Paper-PDF-critical.svg?logo=adobeacrobatreader" alt="Paper PDF">
</a>
<a href="./LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</a>
<a href="https://github.com/UnicomAI/LeMiCa/stargazers" target="_blank">
  <img src="https://img.shields.io/github/stars/UnicomAI/LeMiCa.svg?style=social" alt="GitHub Stars">
</a>

</h5>


<!-- <p align="center" style="background-color:#fafafa; padding:10px; border-radius:10px; border:1px solid #ddd; display:inline-block;">
  <img 
    src="./assets/1_overview.jpg" 
    alt="LeMiCa Overview" 
    width="60%" 
    style="border-radius:6px; box-shadow:0 0 6px rgba(0,0,0,0.1);"
  >
  <br>
  <span style="font-size:0.85em; color:#666;">Figure 1. Overview of LeMiCa framework.</span>
</p> -->


## Introduction

<div align="center">
  ![LeMiCa Overview]("./assets/1_overview.jpg"){ width=60% }
</div>

**LeMiCa** is a training-free acceleration framework for diffusion-based video generation (and extendable to image generation).
Instead of using local heuristic thresholds, LeMiCa formulates cache scheduling as a global path optimization problem
with error-weighted edges and introduces a **Lexicographic Minimax** strategy to bound the worst-case global error.
This global planning improves both inference speed and consistency across frames.
For more details and visual results, please visit our [project page](https://unicomai.github.io/LeMiCa/).


---

## 🔥 Latest News
- [2025/10/20] 🔥 Added experimental support for **Qwen-Image** (Text-to-Image).  
- [2025/09/18] ✨ Selected as a **NeurIPS 2025 Spotlight** paper.  
- [2025/09/18] ✨ Initial public release of LeMiCa. 

---
##  Demos
##### Qwen-Image


<div style="width:85%;max-width:1000px;margin:0 auto;">
  <!-- 图片：无边框，宽度与上面表头一致 -->
  <img
    src="./assets/qw-image.jpg"
    alt="Qwen-Image visual result"
    style="width:100%;height:auto;display:block;margin:10px auto 4px auto;"
  />
</div>

---
##  Supported Models
LeMiCa currently supports and has been tested on the following diffusion-based models:  

**Text-to-Video**
- [Open-Sora](https://github.com/hpcaitech/Open-Sora)  
- [Latte](https://github.com/Vchitect/Latte)  
- [CogVideoX 1.5](https://github.com/THUDM/CogVideo)  
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)  

**Text-to-Image**
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)  

- [FLUX.1](https://github.com/black-forest-labs/flux) 

---

## 🧩 ToDo List
- ✅ Public Project Page  
- ☐ Paper Released  
- ☐ Text-to-Image Forward Inference  
- ☐ Text-to-Video Forward Inference  
- ☐ DAG Construction Code  
- ☐ Open-Source Acceleration Framework   

---

## Acknowledgement
This repository is built based on or inspired by the following open-source projects and diffusion frameworks:  
[Diffusers](https://github.com/huggingface/diffusers),  
[Qwen-Image](https://github.com/QwenLM/Qwen-Image),  
[TeaCache](https://github.com/ali-vilab/TeaCache),  
[VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys).  
We sincerely thank these communities for their open contributions and inspiration.

---

## License
The majority of this project is released under the **Apache 2.0 license** as found in the [LICENSE](./LICENSE) file.

---

## 📖 Citation
If you find **LeMiCa** useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:

```bibtex
@inproceedings{gao2025lemica,
  title     = {LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation},
  author    = {Huanlin Gao and Ping Chen and Fuyuan Shi and Chao Tan and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://unicomai.github.io/LeMiCa/}
}
