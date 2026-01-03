<div align="center">

<h1>Degradation-Aware Metric Prompting for Hyperspectral Image Restoration</h1>

Binfeng Wang<sup>1,3</sup>, Di Wang<sup>2,3</sup>, Haonan Guo<sup>2,3 †</sup>, Ying Fu<sup>1 †</sup>, Jing Zhang<sup>2,3 †</sup>.

<sup>1</sup> Beijing Institute of Technology,  <sup>2</sup> Wuhan University,  <sup>3</sup> Zhongguancun Academy.

<sup>†</sup> Corresponding authors.

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-models">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >

## 🔥 Update

**2026.01.01**
- The paper is post on arXiv! **([arXiv](https://arxiv.org/abs/2512.20251))** 

## 🌞 Abstract

Unified hyperspectral image (HSI) restoration aims to recover various degraded HSIs using a single model, offering great practical value. However, existing methods often depend on explicit degradation priors (e.g., degradation labels) as prompts to guide restoration, which are difficult to obtain due to complex and mixed degradations in real-world scenarios. To address this challenge, we propose a Degradation-Aware Metric Prompting (DAMP) framework. Instead of relying on predefined degradation priors, we design spatial–spectral degradation metrics to continuously quantify multi-dimensional degradations, serving as Degradation Prompts (DP). These DP enable the model to capture cross-task similarities in degradation distributions and enhance shared feature learning. Furthermore, we introduce a Spatial–Spectral Adaptive Module (SSAM) that dynamically modulates spatial and spectral feature extraction through learnable parameters. By integrating SSAM as experts within a Mixture-of-Experts architecture, and using DP as the gating router, the framework enables adaptive, efficient, and robust restoration under diverse, mixed, or unseen degradations. Extensive experiments on natural and remote sensing HSI datasets show that DAMP achieves state-of-the-art performance and demonstrates exceptional generalization capability.

<figure>
<div align="center">
<img src=figs/model.png width="100%">
</div>

**Figure 1. (a) The architecture of the proposed DAMP framework. (b) The Degradation-Adaptive MoE.**

<figure>
<div align="center">
<img src=figs/deg_any.png width="100%">
</div>

**Figure 2. (a) Comparison between explicit prompt-based methods and degradation-aware metric prompting approaches. (b) Confusion matrix for classifying five degradation types based on HFER, STU and SCM. (c) Distribution of different degradation types across the HFER, STU and SCM.**

<figure>
<div align="center">
<img src=figs/res.png width="60%">
</div>

Figure 3. PSNR comparison with the state-of-the-art all-in-one methods: Inpainting, Super Resolution, Gaussian Deblurring, and Gaussian Denoising results are evaluated on the ARAD dataset after unified training, while Poisson Denoising and Motion Deblurring are reported as zero-shot results on the CAVE dataset. $[\cdot]$ denotes the range of PSNR values across different methods.

## 🔨 Usage

Coming Soon.

</div>



## ⭐ Citation

Coming Soon.

## 🎺 Statement

For any other questions please contact Bindeng Wabg at wbf_bit@163.com.
