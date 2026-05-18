import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import random

def main():
    # ================= 1. 环境与模型初始化 =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 正在使用设备: {device}")
    print("⏳ 正在加载预训练的 Stable Diffusion v1.5 模型 (这可能需要几分钟)...")
    
    # 加载 SD 1.5 基座模型
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # 强制使用 DDIM 调度器以保证确定性采样 (呼应论文 2.3 节)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # ================= 2. 实验参数设置 =================
    prompt = "A beautiful landscape of a mountain lake at sunset, highly detailed, 4k resolution"
    seed = 42  # 固定随机种子，保证每次生成的初始噪声一样
    alpha = 0.5  # 论文第三章核心参数：Margin 边界裕度
    payload_size = 512  # 模拟嵌入的机密信息长度 (比特)

    generator = torch.Generator(device).manual_seed(seed)

    # ================= 3. 潜空间初始化与符号调制 (核心创新点) =================
    print("✨ 正在生成初始高斯噪声...")
    # SD 的 VAE 压缩率为 8，所以 512x512 图像的潜变量大小为 64x64，通道数为 4
    latent_shape = (1, pipe.unet.config.in_channels, 64, 64)
    clean_latents = torch.randn(latent_shape, generator=generator, device=device, dtype=pipe.dtype)

    # 复制一份噪声用于隐写调制
    stego_latents = clean_latents.clone()
    
    # 展平潜变量以便于按位置修改
    flat_latents = stego_latents.view(-1)
    
    # 模拟外部输入的机密比特流 (0和1组成的序列)
    np.random.seed(seed)
    secret_bits = np.random.randint(0, 2, payload_size)
    
    print(f"🔐 正在利用 Margin (alpha={alpha}) 机制进行潜空间符号调制...")
    # 选定前 payload_size 个位置进行调制 (论文 3.3.2 节公式实现)
    for i in range(payload_size):
        if secret_bits[i] == 1:
            # 如果是 1，强制推到 alpha 及以上
            flat_latents[i] = torch.clamp(flat_latents[i], min=alpha)
        else:
            # 如果是 0，强制推到 -alpha 及以下
            flat_latents[i] = torch.clamp(flat_latents[i], max=-alpha)
            
    # 重塑回原始维度
    stego_latents = flat_latents.view(latent_shape)

    # ================= 4. 生成与保存图像 =================
    print("🎨 正在生成正常图像 (Clean)...")
    clean_image = pipe(
        prompt=prompt, 
        latents=clean_latents, 
        num_inference_steps=50, 
        guidance_scale=7.5
    ).images[0]

    print("🎨 正在生成含密图像 (Stego)...")
    stego_image = pipe(
        prompt=prompt, 
        latents=stego_latents, 
        num_inference_steps=50, 
        guidance_scale=7.5
    ).images[0]

    # 保存图片
    clean_image.save("clean_sample.png")
    stego_image.save("stego_sample.png")
    print("✅ 成功！图像已保存为 'clean_sample.png' 和 'stego_sample.png'")

if __name__ == "__main__":
    main()