import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：第四章定性分析 (视觉残差热力图生成) ==========")

SECRET = "2022101196"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
MARGIN = 0.65        # 黄金信号强度
RS_REDUNDANCY = 16   # 安全纠错容量
STEPS = 50      
KEY = 12345

# 🎨 挑出3组最具代表性的测试集（用于定性展示）
VIS_PROMPTS = [
    # 1. 风景（纹理复杂，有利于隐藏）
    "A stunning sunset over the Great Wall of China, cinematic lighting",
    # 2. 动物（含有高频细节纹理）
    "A cute cat sitting on a windowsill in an oil painting style",
    # 3. 科幻（色彩鲜艳，具有挑战性）
    "A futuristic laboratory with glowing blue lights and robots"
]

os.makedirs("thesis_qualitative_results", exist_ok=True)
rs = RSCodec(RS_REDUNDANCY)
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 信息预处理
encoded_msg = rs.encode(SECRET.encode('utf-8'))
binary_secret = ''.join(format(b, '08b') for b in encoded_msg)

final_composite_images = []

for i, prompt in enumerate(VIS_PROMPTS):
    print(f"\n>>> 正在生成第 {i+1}/3 组对比图...")
    seed = 2000 + i # 使用不同的种子
    
    # --- 1. 生成原图 ---
    gen_ref = torch.Generator("cuda").manual_seed(seed)
    lat_pure = torch.randn((1, 4, 64, 64), generator=gen_ref, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        img_ref_pil = pipe(prompt=prompt, latents=lat_pure, num_inference_steps=STEPS, guidance_scale=1.0).images[0]
    
    # --- 2. 潜空间嵌入与生成含密图 ---
    flat_lat = lat_pure.clone().view(-1)
    torch.manual_seed(KEY)
    indices = torch.randperm(flat_lat.shape[0])
    for j, bit in enumerate(binary_secret):
        idx = indices[j]
        orig = flat_lat[idx].item()
        if bit == '1':
            flat_lat[idx] = max(orig, MARGIN) if orig > 0 else MARGIN
        else:
            flat_lat[idx] = min(orig, -MARGIN) if orig < 0 else -MARGIN
            
    with torch.no_grad():
        img_stego_pil = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), num_inference_steps=STEPS, guidance_scale=1.0).images[0]
    
    # --- 3. 计算并放大视觉残差 ---
    ref_np = np.array(img_ref_pil).astype(np.int32)
    stego_np = np.array(img_stego_pil).astype(np.int32)
    
    # 计算绝对像素差 (Residual)
    residual_np = np.abs(ref_np - stego_np).astype(np.uint8)
    
    # 将残差平均到单通道 (Luminance difference)
    residual_gray = np.mean(residual_np, axis=2).astype(np.uint8)
    
    # 🌟 关键动作：暴力放大视觉残差 (x50倍)，让微小修改显形
    residual_amplified = (residual_gray.astype(np.float32) * 50.0)
    residual_amplified = np.clip(residual_amplified, 0, 255).astype(np.uint8)
    
    # 使用Matplotlib将单通道灰度图映射为 'hot' 调色的热力图
    cmap = plt.get_cmap('hot')
    heatmap_colored = (cmap(residual_amplified)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored)
    
    # --- 4. 横向拼接三张图 (原图, 含密图, 放大残差) ---
    width, height = img_ref_pil.size
    composite = Image.new('RGB', (width * 3, height))
    composite.paste(img_ref_pil, (0, 0))
    composite.paste(img_stego_pil, (width, 0))
    composite.paste(heatmap_pil, (width * 2, 0))
    
    composite.save(f"thesis_qualitative_results/group_{i+1}_composite.png")
    final_composite_images.append(composite)

print(f"\n🏆 实验结束，视觉残差图已生成在 thesis_qualitative_results 文件夹内。")