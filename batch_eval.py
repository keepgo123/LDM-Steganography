import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
import numpy as np
import pandas as pd
import os
import gc
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

print("========== 毕业设计：大规模自动化测试与画质评估模块 ==========")

# --- 1. 创建结果保存目录 ---
OUTPUT_DIR = "thesis_results_batch"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clean_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "stego_images"), exist_ok=True)

# --- 2. 加载模型 ---
print("正在加载 Stable Diffusion 模型...")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    variant="fp16",
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# --- 3. 核心参数 (与咱们调好的终极防御参数保持绝对一致) ---
STEPS = 50
KEY = 12345
GUIDANCE_SCALE = 3.0  
MARGIN = 1.4          
MSG_LEN = 60   
ECC_LEN = 140  
TOTAL_BYTES = MSG_LEN + ECC_LEN

# 模拟 COCO 数据集中的 Prompt 子集
test_prompts = [
    "A majestic lion resting in the savannah, highly detailed, 4k",
    "A vintage red sports car parked on a city street at night",
    "A cute golden retriever playing with a frisbee in the park",
    "A cozy living room with a fireplace and modern furniture",
    "A delicious pepperoni pizza on a wooden table, food photography"
]

# 模拟批量不同的秘密信息
secret_messages = [
    "Test Message 1: Hello World!",
    "Test Message 2: Stealth is key.",
    "Test Message 3: Diffusion models rock.",
    "Test Message 4: Security verified.",
    "Test Message 5: Graduation defense prep."
]

results = []

print(f"开始批量跑测，共计 {len(test_prompts)} 组样本...")

for idx, (prompt, secret_text) in enumerate(zip(test_prompts, secret_messages)):
    print(f"\n>>> 正在处理样本 [{idx+1}/{len(test_prompts)}]")
    print(f"    Prompt: {prompt[:30]}...")
    
    # 设定固定的随机种子，保证每次生成的初始噪声一致，这样才有对比意义
    SEED = 202605 + idx
    
    # ------------------ A. 生成原始纯净图像 (Clean) ------------------
    gen_clean = torch.Generator("cuda").manual_seed(SEED)
    clean_latents = torch.randn((1, 4, 64, 64), generator=gen_clean, device="cuda", dtype=torch.float16)
    
    clean_img = pipe(prompt=prompt, latents=clean_latents.clone(), guidance_scale=GUIDANCE_SCALE).images[0]
    clean_img.save(os.path.join(OUTPUT_DIR, "clean_images", f"clean_{idx}.png"))
    
    # ------------------ B. 生成含密图像 (Stego) ------------------
    secret_bytes = secret_text.encode('utf-8')
    secret_bytes = secret_bytes.ljust(MSG_LEN, b'\0')
    rs = RSCodec(ECC_LEN)
    encoded_msg = rs.encode(secret_bytes)
    
    flat_lat = clean_latents.clone().view(-1)
    torch.manual_seed(KEY)
    indices = torch.randperm(flat_lat.shape[0])
    binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
    
    for j, bit in enumerate(binary_secret):
        flat_lat[indices[j]] = MARGIN if bit == '1' else -MARGIN
        
    stego_img = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), guidance_scale=GUIDANCE_SCALE).images[0]
    stego_img.save(os.path.join(OUTPUT_DIR, "stego_images", f"stego_{idx}.png"))
    
    # ------------------ C. 图像质量评估 (PSNR & SSIM) ------------------
    img1 = np.array(clean_img)
    img2 = np.array(stego_img)
    
    # 计算 PSNR
    psnr_val = calculate_psnr(img1, img2)
    # 计算 SSIM (针对多通道 RGB 图像需加上 channel_axis=-1)
    ssim_val = calculate_ssim(img1, img2, channel_axis=-1)
    
    print(f"    画质评估 -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    
    # ------------------ D. 提取验证 (确保批量生成不是花架子) ------------------
    try:
        img_tensor = (torch.from_numpy(np.array(stego_img)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
        with torch.no_grad():
            z0 = pipe.vae.encode(img_tensor).latent_dist.mean * pipe.vae.config.scaling_factor
        
        pipe.scheduler.set_timesteps(STEPS, device="cuda")
        
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_emb = pipe.text_encoder(text_input.input_ids.to("cuda"))[0]
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_emb = pipe.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        context = torch.cat([uncond_emb, text_emb])
        
        inv_lat = z0.clone()
        timesteps = pipe.scheduler.timesteps.flip(0)
        
        for i, t in enumerate(timesteps):
            with torch.no_grad():
                latent_model_input = torch.cat([inv_lat] * 2)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=context).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

            prev_timestep = timesteps[i-1] if i > 0 else t 
            alpha_t = pipe.scheduler.alphas_cumprod[t]
            alpha_prev = pipe.scheduler.alphas_cumprod[prev_timestep] if i > 0 else alpha_t
            pred_x0 = (inv_lat - (1 - alpha_t)**0.5 * noise_pred) / alpha_t**0.5
            
            next_t = t + (1000//STEPS) if t < 950 else torch.tensor(999) 
            if next_t < 1000:
                alpha_next = pipe.scheduler.alphas_cumprod[next_t]
            else:
                alpha_next = torch.tensor(1.0, device="cuda", dtype=torch.float16)
                
            inv_lat = alpha_next**0.5 * pred_x0 + (1 - alpha_next)**0.5 * noise_pred

        flat_inv = inv_lat.view(-1)
        torch.manual_seed(KEY)
        extract_indices = torch.randperm(flat_inv.shape[0])
        
        max_bits = TOTAL_BYTES * 8 
        ext_bits = ''.join(['1' if flat_inv[extract_indices[j]] > 0 else '0' for j in range(max_bits)])
        ext_bytes = bytearray([int(ext_bits[j:j+8], 2) for j in range(0, len(ext_bits), 8)])
        
        rs = RSCodec(ECC_LEN)
        decoded, _, _ = rs.decode(ext_bytes)
        extracted_text = decoded.strip(b'\0').decode('utf-8', errors='ignore')
        
        extract_success = (extracted_text == secret_text)
        print(f"    提取验证 -> {'成功 ✅' if extract_success else '失败 ❌'}")
        
    except Exception as e:
        extract_success = False
        print(f"    提取验证 -> 失败 ❌ (Error: {e})")
        
    # 记录这组数据
    results.append({
        "Image_ID": idx,
        "Prompt_Snippet": prompt[:20],
        "PSNR (dB)": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "Extraction_Success": extract_success
    })
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()

# --- 5. 保存量化结果到 CSV 表格 ---
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "quantitative_results.csv")
df.to_csv(csv_path, index=False)

print("\n========== 自动化测试完成！==========")
print(f"总计测试样本: {len(df)}")
print(f"平均 PSNR: {df['PSNR (dB)'].mean():.2f} dB")
print(f"平均 SSIM: {df['SSIM'].mean():.4f}")
print(f"提取成功率: {(df['Extraction_Success'].sum() / len(df)) * 100:.2f}%")
print(f"所有生成的图片和数据表格已保存在 '{OUTPUT_DIR}' 文件夹中。拿着这个 CSV 去画图写论文吧！")