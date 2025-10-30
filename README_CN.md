<a href="./README.md">📚English </a> | 📚中文阅读 

<div align="center">
<!-- <img src="https://github.com/user-attachments/assets/6ceb4269-a861-4545-84db-bad322592156" style="width:auto; height:120px;" />&nbsp; -->
<img src="./docs/static/images/logv2.png" style="width:auto; height:150px;">
</div>


# [NeurIPS 2025 Spotlight] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block">
    <a href="https://github.com/joelulu" target="_blank">高焕霖</a><sup>1,2</sup><sup>*</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&user=gpNOW2UAAAAJ" target="_blank">陈平</a><sup>1,2</sup><sup>*</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/stone002" target="_blank">石芙源</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/tanchaow" target="_blank">谭超</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=en&user=L4OXOs0AAAAJ" target="_blank">刘兆祥</a><sup>1,2</sup>
  </span>
  <br>
  <span class="author-block">
    <a href="https://github.com/FangGet" target="_blank">赵放</a><sup>1,2</sup><sup>†</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CFUQLCAAAAAJ&hl=en" target="_blank">王恺</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com.hk/citations?user=kCC2oKwAAAAJ&hl=zh-CN&oi=ao" target="_blank">廉士国</a><sup>1,2</sup>
  </span>
</div>

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block"><sup>1</sup>中国联通数据科学与人工智能研究院&nbsp;</span>
  <span class="author-block"><sup>2</sup>联通数据智能有限公司</span>
</div>

<div class="is-size-5 publication-authors" align="center">
  (* 共同一作. † 通讯作者.)
</div>

<h5 align="center">

<a href="https://unicomai.github.io/LeMiCa/" target="_blank">
  <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Page">
</a>
<a href="https://github.com/UnicomAI/LeMiCa" target="_blank">
  <img src="https://img.shields.io/badge/Code-GitHub-black.svg?logo=github" alt="Code">
</a>
<a href="https://github.com/UnicomAI/LeMiCa/raw/main/assets/LeMiCa_NeurIPS2025.pdf" target="_blank">
  <img src="https://img.shields.io/badge/Paper-PDF-critical.svg?logo=adobeacrobatreader" alt="Paper PDF">
</a>
<a href="https://github.com/UnicomAI/LeMiCa/raw/main/assets/LeMiCa_NeurIPS2025_appendix.pdf" target="_blank">
  <img src="https://img.shields.io/badge/Appendix-PDF-green.svg?logo=file-pdf" alt="Appendix PDF">
</a>
<a href="./LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</a>
<a href="https://github.com/UnicomAI/LeMiCa/stargazers" target="_blank">
  <img src="https://img.shields.io/github/stars/UnicomAI/LeMiCa.svg?style=social" alt="GitHub Stars">
</a>

</h5>


![LeMiCa Overview](./assets/1_overview_clip.jpg)



## 简介

**LeMiCa** 是一个无需训练的扩散视频生成模型加速算法（也可扩展至图像生成）。不同于以往基于局部启发式阈值的方法，LeMiCa将缓存调度问题表述为带有误差加权边的全局路径优化问题，并引入了词典序极小极大（Lexicographic Minimax）策略，以限制最坏情况下的全局误差。该全局规划方法同时提升了推理速度和跨帧一致性。更多细节与可视化结果，请访问我们的 [项目主页](https://unicomai.github.io/LeMiCa/)。


## 🔥 最近更新
- [2025/10/29] 🚀 代码即将发布，敬请期待！ 
- [2025/10/20] 🔥 新增对文生图模型【**Qwen-Image**】的支持.  
- [2025/09/18] ✨ 论文被选为**NeurIPS 2025 Spotlight**.  
- [2025/09/18] ✨ LeMiCa首次公开发布. 

##  展示
### Open-Sora

https://github.com/user-attachments/assets/ba205856-2d77-494a-aaa9-09189ba2915c


### Qwen-Image

<div style="width:85%;max-width:1000px;margin:0 auto;">
  <!-- 图片：无边框，宽度与上面表头一致 -->
  <img
    src="./assets/qw-image.jpg"
    alt="Qwen-Image visual result"
    style="width:100%;height:auto;display:block;margin:10px auto 4px auto;"
  />
</div>


##  支持模型列表
LeMiCa 目前支持并已在以下基于扩散的模型上进行了测试：  

**文生视频**
- [Open-Sora](https://github.com/hpcaitech/Open-Sora)  
- [Latte](https://github.com/Vchitect/Latte)  
- [CogVideoX 1.5](https://github.com/THUDM/CogVideo)  
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)  

**文生图**
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)  
- [FLUX.1](https://github.com/black-forest-labs/flux) 



## 🧩 待办列表
- ✅ 公开项目主页  
- ✅ 发布论文  
- ☐ 文生图的前向推理 
- ☐ 文生视频的前向推理  
- ☐ DAG建图代码 
- ☐ 开源通用加速框架   



## 致谢
本仓库基于或受到以下开源项目的启发：[Diffusers](https://github.com/huggingface/diffusers)、[Qwen-Image](https://github.com/QwenLM/Qwen-Image)、[TeaCache](https://github.com/ali-vilab/TeaCache)、[VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)。我们衷心感谢这些社区的开放贡献与启发。



## 许可协议
本项目的大部分内容依据 [LICENSE](./LICENSE) 文件中的**Apache 2.0 许可协议**发布。

## 📖 引用
如果您在研究或应用中发现 **LeMiCa** 有所帮助，请考虑为我们点⭐并通过以下BibTeX条目引用：


```bibtex
@inproceedings{gao2025lemica,
  title     = {LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation},
  author    = {Huanlin Gao and Ping Chen and Fuyuan Shi and Chao Tan and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://unicomai.github.io/LeMiCa/}
}
