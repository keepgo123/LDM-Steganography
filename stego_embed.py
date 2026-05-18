import torch
from diffusers import StableDiffusionPipeline

print("--- 终极演示版：自然风景隐写测试 ---")

# ==========================================
# 1. 基础设置（换成了风景主题）
# ==========================================
# 提示词：雪山中温馨的木屋，日落时分
PROMPT = "A beautiful cozy wooden cabin in a snowy mountain landscape, sunset, highly detailed, 4k resolution, photorealistic"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
SEED = 2026  # 换一个全新的随机种子
SECRET = "2022101196" # 你的学号

# 学号转二进制
binary_secret = ''.join(format(ord(i), '08b') for i in SECRET)

# ==========================================
# 2. 加载模型
# ==========================================
print("正在加载模型...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")

# ==========================================
# 3. 生成初始噪声并进行“无损符号调制”
# ==========================================
print("正在生成初始噪声并注入学号信息...")
generator = torch.Generator("cuda").manual_seed(SEED)
latents = torch.randn(
    (1, 4, 64, 64), 
    generator=generator, 
    device="cuda", 
    dtype=torch.float16
)

flat_latents = latents.view(-1) 
total_length = flat_latents.shape[0]

# 使用密钥 12345 随机打乱位置，避免斑块聚集
torch.manual_seed(12345) 
permuted_indices = torch.randperm(total_length)

# 纯符号翻转嵌入（只改正负号，不改大小）
for i, bit in enumerate(binary_secret):
    idx = permuted_indices[i]
    current_val = abs(flat_latents[idx])
    
    if bit == '1':
        flat_latents[idx] = current_val
    else:
        flat_latents[idx] = -current_val

# 恢复矩阵形状
latents = flat_latents.view(1, 4, 64, 64)

# ==========================================
# 4. 生成图像
# ==========================================
print("正在生成风景图像（采样 50 步，请稍候）...")
with torch.no_grad():
    image = pipe(
        prompt=PROMPT,
        latents=latents, 
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

filename = "stego_landscape_secret.png"
image.save(filename)
print(f"大功告成！含密风景图已保存为 {filename}")