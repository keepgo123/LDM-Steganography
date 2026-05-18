import torch
from diffusers import StableDiffusionPipeline
import numpy as np

print("--- 应急计划：开始生成正常的含密猫咪图片 ---")

# ==========================================
# 参数设置（经过验证的稳定参数）
# ==========================================
ALPHA = 0.05       # 嵌入强度：0.05是一个平衡点，既隐形又保质
STEPS = 50         # 增加到50步采样，确保模型有足够空间修正结构
PROMPT = "a majestic cat sitting on a throne, high resolution, photorealistic"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
SEED = 42
SECRET = "2022101196" # 你的学号！

# ==========================================
# 第一步：秘密信息预处理
# ==========================================
print(f"1. 将学号 '{SECRET}' 转换为比特流...")
binary_secret = ''.join(format(ord(i), '08b') for i in SECRET)
num_bits = len(binary_secret)
print(f"   准备嵌入 {num_bits} 个比特。")

# ==========================================
# 第二步：加载模型
# ==========================================
print(f"\n2. 加载 Stable Diffusion 模型 (CUDA)...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16
).to("cuda")

# ==========================================
# 第三步：生成初始噪声并进行“温和”嵌入
# ==========================================
print(f"\n3. 生成初始噪声(Seed={SEED})并嵌入秘密信息 (ALPHA={ALPHA})...")
generator = torch.Generator("cuda").manual_seed(SEED)
latents = torch.randn(
    (1, 4, 64, 64), 
    generator=generator, 
    device="cuda", 
    dtype=torch.float16
)

# 展平矩阵方便修改
flat_latents = latents.view(-1) 

# --- 温和嵌入逻辑：只微调数值，不剧烈改变分布 ---
# 我们使用一个在隐写论文中常见的简单公式：
# Modified_Noise = Alpha * Bit + (1-Alpha) * Noise
# 这里Bit是 {-1, 1}，Noise是 {randn}
for i, bit in enumerate(binary_secret):
    # 将 0/1 比特映射到 -1/1
    symbol = 1.0 if bit == '1' else -1.0
    # 这一行代码是温和嵌入的核心
    flat_latents[i] = ALPHA * symbol + (1 - ALPHA) * flat_latents[i]

# 恢复矩阵形状
latents = flat_latents.view(1, 4, 64, 64)
print("   信息注入完毕。")

# ==========================================
# 第四步：用带着秘密信息的噪声，生成图片
# ==========================================
print(f"\n4. 正在生成带有你学号的猫咪图片 (采样 {STEPS} 步)...")
with torch.no_grad():
    image = pipe(
        prompt=PROMPT,
        latents=latents, 
        num_inference_steps=STEPS,
        guidance_scale=7.5 # 标准引导比例
    ).images[0]

# 保存
filename = "stego_cat_final_normal.png"
image.save(filename)
print(f"\n大功告成！含密图片已保存为 {filename}")
print("这次请仔细检查：图片中的猫，是否恢复正常了？")