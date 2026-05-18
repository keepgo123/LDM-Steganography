import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import pandas as pd
import math
import gc
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：核心参数消融实验 (Margin 质量与鲁棒性折中曲线) ==========")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
STEPS = 50
KEY = 12345
SECRET = "2022101196" # 10字节学号

# 我们要测试的 Margin 梯度
margins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompt = "A cute cat sitting on a windowsill in an oil painting style"
seed = 1024 

# 将秘密信息转为纯二进制比特流
binary_secret = ''.join(format(ord(c), '08b') for c in SECRET)
num_bits = len(binary_secret)

# 1. 先生成一张完全不含密的【纯净原图】，用于计算 PSNR
print(">>> 正在生成纯净基准原图...")
gen = torch.Generator("cuda").manual_seed(seed)
clean_latents = torch.randn((1, 4, 64, 64), generator=gen, device="cuda", dtype=torch.float16)
clean_img = pipe(prompt=prompt, latents=clean_latents.clone(), guidance_scale=1.0).images[0]
clean_np = np.array(clean_img).astype(np.float32)

results = []

for m in margins:
    print(f"\n>>> 正在测试 Margin = {m}...")
    
    # 2. 嵌入数据
    flat_lat = clean_latents.clone().view(-1)
    torch.manual_seed(KEY)
    indices = torch.randperm(flat_lat.shape[0])
    
    for j, bit in enumerate(binary_secret):
        idx = indices[j]
        flat_lat[idx] = m if bit == '1' else -m
        
    stego_img = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), guidance_scale=1.0).images[0]
    stego_np = np.array(stego_img).astype(np.float32)
    
    # 3. 计算 PSNR (图像质量)
    mse = np.mean((clean_np - stego_np) ** 2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 100.0
    
    # 4. 提取数据
    img_tensor = (torch.from_numpy(np.array(stego_img)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
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

    # 5. 计算比特误码率 (BER)
    flat_inv = inv_lat.view(-1)
    torch.manual_seed(KEY)
    extract_indices = torch.randperm(flat_inv.shape[0])
    
    bit_errors = 0
    for j in range(num_bits):
        ext_bit = '1' if flat_inv[extract_indices[j]] > 0 else '0'
        if ext_bit != binary_secret[j]:
            bit_errors += 1
            
    ber = (bit_errors / num_bits) * 100
    
    print(f"    图像质量 (PSNR): {psnr:.2f} dB")
    print(f"    底层比特误码率 (BER): {ber:.2f}%")
    
    results.append({
        "Margin": m, 
        "PSNR (dB)": round(psnr, 2), 
        "BER (%)": round(ber, 2)
    })

    # 清理显存
    try:
        del img_tensor, z0, inv_lat, text_emb, noise_pred
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()

# 保存结果
df = pd.DataFrame(results)
df.to_csv("thesis_quantitative_results/ablation_margin.csv", index=False)
print("\n🏆 参数消融实验完成！数据已保存至 CSV。")