import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import gc
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：第四章鲁棒性测试 (抗攻击实验 - 完美优化版) ==========")

SECRET = "2022101196"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
STEPS = 50
RS_REDUNDANCY = 24  # 🌟 提升纠错冗余度，最高可纠错 12 字节
KEY = 12345

# 加载模型（关闭 safety_checker 避免卡死并节省内存）
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

rs = RSCodec(RS_REDUNDANCY)
encoded_msg = rs.encode(SECRET.encode('utf-8'))
num_bits = len(encoded_msg) * 8

# 定义攻击函数
def apply_attack(img_pil, attack_type):
    img_np = np.array(img_pil)
    if attack_type == "None":
        return img_pil
    
    elif attack_type == "JPEG_Compression":
        # 模拟微信级别压缩 (Quality=80)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=80)
        return Image.open(buffer)
    
    elif attack_type == "Gaussian_Noise":
        # 🌟 调整为更符合真实信道的高斯底噪 (scale=3)
        noise = np.random.normal(0, 3, img_np.shape).astype(np.int16)
        img_noised = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_noised)

    elif attack_type == "Resizing":
        # 缩小到 400x400 再拉伸回 512x512
        small = img_pil.resize((400, 400), Image.BILINEAR)
        return small.resize((512, 512), Image.BILINEAR)

# 测试流程
attack_list = ["None", "JPEG_Compression", "Gaussian_Noise", "Resizing"]
prompt = "A cute cat sitting on a windowsill in an oil painting style"
seed = 1002 

# 1. 预先生成一张基础含密图
print(f">>> 正在准备基础含密图...")
gen = torch.Generator("cuda").manual_seed(seed)
latents = torch.randn((1, 4, 64, 64), generator=gen, device="cuda", dtype=torch.float16)

# 嵌入逻辑 
flat_lat = latents.clone().view(-1)
torch.manual_seed(KEY)
indices = torch.randperm(flat_lat.shape[0])
binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
for j, bit in enumerate(binary_secret):
    idx = indices[j]
    flat_lat[idx] = 0.65 if bit == '1' else -0.65
stego_img_base = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), guidance_scale=1.0).images[0]

results = []

for attack in attack_list:
    print(f">>> 正在测试攻击类型: {attack}...")
    attacked_img = apply_attack(stego_img_base, attack)
    
    # 提取逻辑 (🌟 加上 torch.no_grad() 防止 OOM 显存溢出)
    img_tensor = (torch.from_numpy(np.array(attacked_img)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
    with torch.no_grad():
        z0 = pipe.vae.encode(img_tensor).latent_dist.mean * pipe.vae.config.scaling_factor
    
    # DDIM Inversion
    pipe.scheduler.set_timesteps(STEPS, device="cuda")
    text_emb = pipe.text_encoder(pipe.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda"))[0]
    inv_lat = z0.clone()
    for t in pipe.scheduler.timesteps.flip(0):
        with torch.no_grad():
            noise_pred = pipe.unet(inv_lat, t, encoder_hidden_states=text_emb).sample
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        next_t = t + (1000//STEPS) if t < 950 else torch.tensor(999) 
        alpha_next = pipe.scheduler.alphas_cumprod[next_t] if next_t < 1000 else torch.tensor(1.0)
        pred_x0 = (inv_lat - (1 - alpha_t)**0.5 * noise_pred) / alpha_t**0.5
        inv_lat = alpha_next**0.5 * pred_x0 + (1 - alpha_next)**0.5 * noise_pred

    # 比特提取
    flat_inv = inv_lat.view(-1)
    torch.manual_seed(KEY)
    extract_indices = torch.randperm(flat_inv.shape[0])
    ext_bits = ''.join(['1' if flat_inv[extract_indices[j]] > 0 else '0' for j in range(num_bits)])
    ext_bytes = bytearray([int(ext_bits[j:j+8], 2) for j in range(0, len(ext_bits), 8)])
    
    # 纠错前误码数
    errors = sum([1 for j in range(len(encoded_msg)) if encoded_msg[j] != ext_bytes[j]])
    
    status = "Failed"
    try:
        decoded, _, _ = rs.decode(ext_bytes)
        if decoded.decode('utf-8') == SECRET: 
            status = "Success"
    except: 
        pass
    
    print(f"    结果: {status} | 原始误码: {errors} 字节")
    results.append({"Attack": attack, "Errors": errors, "Status": status})

    # 🌟 显存清理逻辑，保证连续运行不崩溃
    try:
        del img_tensor, z0, inv_lat, text_emb, noise_pred
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

# 保存结果
pd.DataFrame(results).to_csv("thesis_quantitative_results/robustness_results.csv", index=False)
print("\n🏆 完美版鲁棒性实验完成！数据已保存，全线收工！")