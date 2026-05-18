import torch
from diffusers import StableDiffusionPipeline

print("1. 开始加载扩散模型，首次运行会自动下载约4GB的权重文件，请耐心等待...")
# 加载 Stable Diffusion v1-5 模型
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# 将模型推送到你的高级显卡上全速运行
pipe = pipe.to("cuda")
print("模型加载成功！")

print("2. 正在生成初始潜空间噪声 (Latents)...")
# ⚠️ 注意这里：这个 (1, 4, 64, 64) 的高斯噪声矩阵，就是你接下来几天要“藏”秘密信息的地方！
generator = torch.Generator("cuda").manual_seed(42) 
init_latents = torch.randn(
    (1, 4, 64, 64),
    generator=generator,
    device="cuda",
    dtype=torch.float16
)

print("3. 开始生成图片...")
prompt = "a cute cyber cat wearing a spacesuit, 4k resolution, high detail"
# 把我们自己生成的噪声塞进 pipeline 里
image = pipe(
    prompt=prompt,
    latents=init_latents, 
    num_inference_steps=20 # 采样 20 步
).images[0]

print("4. 正在保存图片...")
image.save("first_test.png")
print("太棒了！初战告捷，图片已保存为 first_test.png")