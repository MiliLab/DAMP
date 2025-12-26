<div align="center">

<h1>SARMAE: Masked Autoencoder for SAR Representation Learning</h1>

Binfeng Wang<sup>1,3</sup>, Di Wang<sup>2,3</sup>, Haonan Guo<sup>2,3 †</sup>, Ying Fu<sup>1 †</sup>, Jing Zhang<sup>2,3 †</sup>.

<sup>1</sup> Beijing Institute of Technology,  <sup>2</sup> Wuhan University,  <sup>3</sup> Zhongguancun Academy.

<sup>†</sup> Corresponding authors.

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-models">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >

## 🔥 Update

**2025.12.25**

- The paper is post on arXiv! **([arXiv DAMP](https://arxiv.org/abs/2512.20251))**

## 🌞 Abstract

Unified hyperspectral image (HSI) restoration aims to recover various degraded HSIs using a single model, offering great practical value. However, existing methods often depend on explicit degradation priors (e.g., degradation labels) as prompts to guide restoration, which are difficult to obtain due to complex and mixed degradations in real-world scenarios. To address this challenge, we propose a Degradation-Aware Metric Prompting (DAMP) framework. Instead of relying on predefined degradation priors, we design spatial–spectral degradation metrics to continuously quantify multi-dimensional degradations, serving as Degradation Prompts (DP). These DP enable the model to capture cross-task similarities in degradation distributions and enhance shared feature learning. Furthermore, we introduce a Spatial–Spectral Adaptive Module (SSAM) that dynamically modulates spatial and spectral feature extraction through learnable parameters. By integrating SSAM as experts within a Mixture-of-Experts architecture, and using DP as the gating router, the framework enables adaptive, efficient, and robust restoration under diverse, mixed, or unseen degradations. Extensive experiments on natural and remote sensing HSI datasets show that DAMP achieves state-of-the-art performance and demonstrates exceptional generalization capability.

<figure>
<div align="center">
<img src=Figs/model.png width="100%">
</div>

<div align='center'>

**Figure 1. Overview of the SARMAE pretraining framework. The framework consists of two branches: (i) a SAR branch following the MAE architecture with Speckle-Aware Representation Enhancement (SARE) to handle inherent speckle noise, and (ii) an optical branch using a frozen DINOv3 encoder. For paired SAR-optical data, Semantic Anchor Representation Constraint (SARC) aligns SAR features with semantic-rich optical representations. Unpaired SAR images are processed solely through the SAR branch.**

</div>

## 📖 Datasets

<figure>
<div align="center">
<img src=Figs/dataset.png width="40%">
</div>

<div align='center'>

**Figure 2. The organization of data sources in SAR-1M.**

</div>

## 🚀 Models

Coming Soon.

## 🔨 Usage

Coming Soon.

## 🍭 Results

<figure>
<div align="center">
<img src=Figs/radar.png width="50%">
</div>

<div align='center'>

**Figure 3. SARMAE outperforms SOTA methods on multiple datasets. <sup>1</sup>: 40-SHOT; <sup>2</sup>: 30% labeled. <sup>a</sup>: Multi-classes; <sup>b</sup>: Water.**

</div>

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2" align="center"><b>FUSAR-SHIP</b></th>
    <th colspan="2" align="center"><b>MSTAR</b></th>
    <th colspan="1" align="center"><b>SAR-ACD</b></th>
  </tr>
  <tr>
    <th align="center">40-shot</th>
    <th align="center">30%</th>
    <th align="center">40-shot</th>
    <th align="center">30%</th>
    <th align="center">30%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ResNet-50</td>
    <td align="center">-</td>
    <td align="center">58.41</td>
    <td align="center">-</td>
    <td align="center">89.94</td>
    <td align="center">59.70</td>
  </tr>
  <tr>
    <td>Swin Transformer</td>
    <td align="center">-</td>
    <td align="center">60.79</td>
    <td align="center">-</td>
    <td align="center">82.97</td>
    <td align="center">67.50</td>
  </tr>
  <tr>
    <td>Bet</td>
    <td align="center">59.70</td>
    <td align="center">71.13</td>
    <td align="center">40.70</td>
    <td align="center">69.75</td>
    <td align="center">79.77</td>
  </tr>
  <tr>
    <td>LoMaR</td>
    <td align="center">82.70</td>
    <td align="center">-</td>
    <td align="center">77.00</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>SAR-JEPA</td>
    <td align="center">85.80</td>
    <td align="center">-</td>
    <td align="center">91.60</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>SUMMIT</td>
    <td align="center">-</td>
    <td align="center">71.91</td>
    <td align="center">-</td>
    <td align="center">98.39</td>
    <td align="center">84.25</td>
  </tr>
  <tr style="border-top: 2px solid #999;">
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center">89.30</td>
    <td align="center">92.92</td>
    <td align="center">96.70</td>
    <td align="center"><b>99.61</b></td>
    <td align="center">95.06</td>
  </tr>
  <tr>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center"><b>90.86</b></td>
    <td align="center">92.80</td>
    <td align="center"><b>97.24</b></td>
    <td align="center">98.92</td>
    <td align="center"><b>95.63</b></td>
  </tr>
</tbody>
</table>

**Table 1.** Performance comparison (Top1 Accuracy, %) of different methods on the target classification task.

</div>

<table>
<thead>
  <tr>
    <th align="center">Method</th>
    <th align="center">SARDet-100k</th>
    <th align="center">SSDD</th>
    <th align="center">Method</th>
    <th align="center">RSAR</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ImageNet</td>
    <td align="center">52.30</td>
    <td align="center">66.40</td>
    <td>RoI Transformer</td>
    <td align="center">35.02</td>
  </tr>
  <tr>
    <td>Deformable DETR</td>
    <td align="center">50.00</td>
    <td align="center">52.60</td>
    <td>Def. DETR</td>
    <td align="center">46.62</td>
  </tr>
  <tr>
    <td>Swin Transformer</td>
    <td align="center">53.80</td>
    <td align="center">40.70</td>
    <td>RetinaNet</td>
    <td align="center">57.67</td>
  </tr>
  <tr>
    <td>ConvNeXt</td>
    <td align="center">55.10</td>
    <td align="center">-</td>
    <td>ARS-DETR</td>
    <td align="center">61.14</td>
  </tr>
  <tr>
    <td>CATNet</td>
    <td align="center">-</td>
    <td align="center">64.66</td>
    <td>R3Det</td>
    <td align="center">63.94</td>
  </tr>
  <tr>
    <td>MSFA</td>
    <td align="center">56.40</td>
    <td align="center">-</td>
    <td>ReDet</td>
    <td align="center">64.71</td>
  </tr>
  <tr>
    <td>SARAFE</td>
    <td align="center">57.30</td>
    <td align="center">67.50</td>
    <td>O-RCNN</td>
    <td align="center">64.82</td>
  </tr>
  <tr style="border-top: 2px solid #999;">
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center">57.90</td>
    <td align="center">68.10</td>
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center">66.80</td>
  </tr>
  <tr>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center"><b>63.10</b></td>
    <td align="center"><b>69.30</b></td>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center"><b>72.20</b></td>
  </tr>
</tbody>
</table>

**Table 2.** Performance comparison (mAP, %) of different methods on horizontal and oriented object detection tasks.

</div>

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="7" align="center"><b>Multiple classes</b></th>
    <th colspan="1" align="center"><b>Water</b></th>
  </tr>
  <tr>
    <th>Industrial Area</th>
    <th>Natural Area</th>
    <th>Land Use</th>
    <th>Water</th>
    <th>Housing</th>
    <th>Other</th>
    <th>mIoU</th>
    <th>IoU</th>
  </tr> 
</thead>
<tbody>
  <tr>
    <td>FCN</td>
    <td align="center">37.78</td>
    <td align="center">71.58</td>
    <td align="center">1.24</td>
    <td align="center">72.76</td>
    <td align="center">67.69</td>
    <td align="center">39.05</td>
    <td align="center">48.35</td>
    <td align="center">85.95</td>
  </tr>
  <tr>
    <td>ANN</td>
    <td align="center">41.23</td>
    <td align="center">72.92</td>
    <td align="center">0.97</td>
    <td align="center">75.95</td>
    <td align="center">68.40</td>
    <td align="center">56.01</td>
    <td align="center">52.58</td>
    <td align="center">87.32</td>
  </tr>
  <tr>
    <td>PSPNet</td>
    <td align="center">33.99</td>
    <td align="center">72.31</td>
    <td align="center">0.93</td>
    <td align="center">76.51</td>
    <td align="center">68.07</td>
    <td align="center">57.07</td>
    <td align="center">51.48</td>
    <td align="center">87.13</td>
  </tr>
  <tr>
    <td>DeepLab V3+</td>
    <td align="center">40.62</td>
    <td align="center">70.67</td>
    <td align="center">0.55</td>
    <td align="center">72.93</td>
    <td align="center">69.96</td>
    <td align="center">34.53</td>
    <td align="center">48.21</td>
    <td align="center">87.53</td>
  </tr>
  <tr>
    <td>PSANet</td>
    <td align="center">40.70</td>
    <td align="center">69.46</td>
    <td align="center">1.33</td>
    <td align="center">69.46</td>
    <td align="center">68.75</td>
    <td align="center">32.68</td>
    <td align="center">47.14</td>
    <td align="center">86.18</td>
  </tr>
  <tr>
    <td>DANet</td>
    <td align="center">39.56</td>
    <td align="center">72.00</td>
    <td align="center">1.00</td>
    <td align="center">74.95</td>
    <td align="center">67.79</td>
    <td align="center">56.28</td>
    <td align="center">39.56</td>
    <td align="center">89.29</td>
  </tr>
  <tr style="border-top: 2px solid #999;">
    <td><b>SARMAE(ViT-B)</b></td>
    <td align="center"><b>65.87</b></td>
    <td align="center">75.65</td>
    <td align="center">29.20</td>
    <td align="center">84.01</td>
    <td align="center">73.23</td>
    <td align="center"><b>71.21</b></td>
    <td align="center">66.53</td>
    <td align="center">92.31</td>
  </tr>
  <tr>
    <td><b>SARMAE(ViT-L)</b></td>
    <td align="center">65.84</td>
    <td align="center"><b>78.04</b></td>
    <td align="center"><b>29.47</b></td>
    <td align="center"><b>87.12</b></td>
    <td align="center"><b>75.22</b></td>
    <td align="center">69.34</td>
    <td align="center"><b>67.51</b></td>
    <td align="center"><b>93.06</b></td>
  </tr>
</tbody>
</table>

**Table 3.** Performance comparison of semantic segmentation methods on multiple classes and water classes.

## ⭐ Citation

If you find SARMAE helpful, please give a ⭐ and cite it as follows:

```
@misc{liu2025sarmaemaskedautoencodersar,
      title={SARMAE: Masked Autoencoder for SAR Representation Learning}, 
      author={Danxu Liu and Di Wang and Hebaixu Wang and Haoyang Chen and Wentao Jiang and Yilin Cheng and Haonan Guo and Wei Cui and Jing Zhang},
      year={2025},
      eprint={2512.16635},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16635}, 
}
```

## 🎺 Statement

For any other questions please contact Danxu Liu at [bit.edu.cn](3120245436@bit.edu.cn) or [gmail.com](ldx.wenquandan@gmail.com).


