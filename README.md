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

**2026.05.05**
- The paper is accepted at ICML2026! **([arXiv](https://arxiv.org/abs/2512.20251))**

**2026.06.15**
- The source code released!

## 🌞 Abstract

Unified hyperspectral image (HSI) restoration aims to recover diverse degradations within a single model. However, current methods often rely on impractical explicit priors or opaque black-box representations that overfit to training distributions, hampering generalization to unseen scenarios. To bridge this gap, we propose Degradation-Aware Metric Prompting (DAMP), a novel framework that characterizes multi-dimensional degradations through interpretable spatial-spectral metrics. These metrics serve as Degradation Prompts (DP), enabling the model to capture shared characteristics across tasks and adapt to unknown corruptions. Central to our framework is the Degradation-Adaptive Mixture-of-Experts (DAMoE), where Spatial-Spectral Adaptive Modules (SSAMs) serve as experts that utilize learnable fusion coefficients to specialize in distinct degradation degrees. By using DP as a gating router, DAMoE dynamically activates specialized experts tailored to the specific degradation profile. Extensive experiments on natural and remote sensing HSI datasets demonstrate that DAMP achieves state-of-the-art performance and exhibits exceptional zero-shot generalization on unseen restoration tasks.

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

```
python main.py
```



## ⭐ Citation

```
@article{wang2025degradation,
  title={Degradation-Aware Metric Prompting for Hyperspectral Image Restoration},
  author={Wang, Binfeng and Wang, Di and Guo, Haonan and Fu, Ying and Zhang, Jing},
  journal={arXiv preprint arXiv:2512.20251},
  year={2025}
}
```

## 🎺 Statement

For any other questions please contact Bindeng Wabg at wbf_bit@163.com.
