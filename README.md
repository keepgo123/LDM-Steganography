# LDM-Based Image Steganography System

基于潜在扩散模型（LDM）的图像隐写系统。

## 项目说明

本项目实现了基于潜在扩散模型（Latent Diffusion Model, LDM）的图像隐写方案，在图像生成过程中将秘密信息嵌入到扩散模型的隐空间表示中，同时保持生成图像的视觉质量。项目包含完整的隐写嵌入与提取流程，以及鲁棒性评估、攻击检测和消融实验等辅助模块。

### 核心思路

利用扩散模型在去噪过程中对隐空间特征的精细控制能力，通过调制特征图的 margin 来实现信息的高容量、高隐蔽性嵌入。提取端利用对应的解码器从生成图像中恢复隐藏信息。

### 功能模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 隐写嵌入 | `stego_embed.py`, `stego_embed_final.py`, `final_system.py` | 将秘密信息嵌入到图像生成过程 |
| 应用演示 | `app_demo.py`, `demo.py` | 图形化演示与交互界面 |
| 鲁棒性测试 | `robustness_test.py`, `robust_stego.py` | JPEG压缩、缩放、噪声等鲁棒性评估 |
| 攻击评估 | `attack_eval.py`, `extra_attacks.py`, `security_eval.py` | 隐写分析攻击与安全性评估 |
| 容量测试 | `capacity_test.py` | 嵌入容量上限测试 |
| 消融实验 | `ablation_margin.py`, `plot_ablation.py`, `draw_ablation.py` | 各模块消融对比实验 |
| 批量评估 | `batch_eval.py`, `final_batch_test.py`, `honest_final_test.py` | 大规模批量测试 |
| 可视化 | `draw_charts.py`, `draw_diffusion.py`, `visualization_heatmap.py` | 结果图表与热力图生成 |
| COCO测试 | `run_coco_500.py` | COCO数据集上的基准测试 |
| 基线生成 | `00_generate_baseline.py` | 生成对比基线 |

## 环境配置

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 嵌入秘密信息
python stego_embed_final.py

# 运行完整系统
python final_system.py

# 启动演示应用
python app_demo.py
```

## 项目声明

- **项目名称**：基于扩散模型的图像隐写系统（LDM-Based Image Steganography System）
- **项目作者**：[你的姓名]
- **作者单位**：[你的学校及学院]
- **开发语言**：Python
- **框架**：Diffusers / Transformers / PyTorch / Gradio / FastAPI
- **核心技术**：潜在扩散模型（LDM）、隐空间特征调制、图像隐写与提取、图像隐私保护

本项目为毕业论文实验代码，仅供学术研究与学习参考。未经作者许可，不得用于商业用途。

如使用本项目代码，请引用论文：

```
@thesis{lm-steganography,
  title     = {基于扩散模型的图像隐写系统},
  author    = {XXX},
  school    = {XX大学},
  year      = {2026},
}
```

## 开源地址

https://github.com/keepgo123/LDM-Steganography

## 开发信息

- **开发语言**：Python
- **代码规模**：约2600行
- **发布时间**：2026年5月
