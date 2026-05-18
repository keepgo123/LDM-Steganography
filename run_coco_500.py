import os
# --- 绝对防御：强制把 Hugging Face 缓存路径改到 D 盘！ ---
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache"

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
import numpy as np
import pandas as pd
import gc
import json
import io
import time
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

print("========== 毕业设计 终极兵器：COCO 500张大样本自动化跑测系统 ==========")

# --- 1. 核心参数设置 ---
STEPS = 50
KEY = 12345
GUIDANCE_SCALE = 1.0  
MARGIN = 1.2         
MSG_LEN = 60   
ECC_LEN = 140  
TOTAL_BYTES = MSG_LEN + ECC_LEN

TARGET_SAMPLES = 500  # 目标测试数量
OUTPUT_DIR = "thesis_final_results"
CSV_PATH = os.path.join(OUTPUT_DIR, "coco_500_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clean_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "stego_images"), exist_ok=True)

# --- 2. 尝试读取真正的 COCO 提示词 ---
coco_prompts = []
coco_json_path = "annotations/captions_val2017.json"
if os.path.exists(coco_json_path):
    print(f"✅ 找到 COCO 数据集: {coco_json_path}")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    coco_prompts = [ann['caption'] for ann in coco_data['annotations']][:TARGET_SAMPLES]
else:
    print(f"⚠️ 未检测到 {coco_json_path}！系统将自动生成备用学术测试 Prompt 列表进行测试...")
    base_prompts = [
        "A photo of a dog playing in the park", "A modern city street with cars at night", 
        "A plate of delicious food on a wooden table", "A majestic mountain landscape with clouds",
        "A close up of a beautiful flower in a garden"
    ]
    coco_prompts = [(base_prompts[i % 5] + f", highly detailed, sharp focus, variant {i}") for i in range(TARGET_SAMPLES)]

# --- 3. 加载模型 ---
print("正在加载 Stable Diffusion 模型 (FP16 加速模式)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16, 
    variant="fp16",
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# --- 4. 断点续传逻辑 ---
results = []
start_idx = 0
if os.path.exists(CSV_PATH):
    try:
        existing_df = pd.read_csv(CSV_PATH)
        results = existing_df.to_dict('records')
        start_idx = len(results)
        print(f"🔄 检测到历史数据！已自动跳过前 {start_idx} 个样本，继续执行任务。")
    except Exception as e:
        print("读取历史数据失败，将重新开始。")

# --- 5. 开启挂机主循环 ---
print(f"\n🚀 开始冲刺！将运行到第 {TARGET_SAMPLES} 个样本...")
start_time = time.time()

for idx in range(start_idx, TARGET_SAMPLES):
    prompt = coco_prompts[idx]
    secret_text = f"Penghui_Graduation_ID_{idx:04d}_" + os.urandom(8).hex()
    
    print(f"\n>>> 正在处理 [{idx+1}/{TARGET_SAMPLES}] | Prompt: {prompt[:30]}...")
    SEED = 20260501 + idx
    
    try:
        # ---- A. 生成 Clean 图像 ----
        gen_clean = torch.Generator("cuda").manual_seed(SEED)
        clean_latents = torch.randn((1, 4, 64, 64), generator=gen_clean, device="cuda", dtype=torch.float16)
        clean_img = pipe(prompt=prompt, latents=clean_latents.clone(), guidance_scale=GUIDANCE_SCALE).images[0]
        if idx < 20: 
            clean_img.save(os.path.join(OUTPUT_DIR, "clean_images", f"clean_{idx:04d}.png"))
        
        # ---- B. 生成 Stego 图像 ----
        secret_bytes = secret_text.encode('utf-8').ljust(MSG_LEN, b'\0')
        rs = RSCodec(ECC_LEN)
        encoded_msg = rs.encode(secret_bytes)
        
        flat_lat = clean_latents.clone().view(-1)
        torch.manual_seed(KEY)
        indices = torch.randperm(flat_lat.shape[0])
        binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
        for j, bit in enumerate(binary_secret):
            flat_lat[indices[j]] = MARGIN if bit == '1' else -MARGIN
            
        stego_img = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), guidance_scale=GUIDANCE_SCALE).images[0]
        if idx < 20:
            stego_img.save(os.path.join(OUTPUT_DIR, "stego_images", f"stego_{idx:04d}.png"))
        
        # ---- C. 计算 PSNR / SSIM ----
        img1 = np.array(clean_img)
        img2 = np.array(stego_img)
        psnr_val = calculate_psnr(img1, img2)
        ssim_val = calculate_ssim(img1, img2, channel_axis=-1)
        
        # ---- D. 极恶劣信道攻击测试 (JPEG 80) ----
        img_byte_arr = io.BytesIO()
        stego_img.save(img_byte_arr, format='JPEG', quality=80)
        attacked_img = Image.open(img_byte_arr).convert("RGB")
        
        # ---- E. 提取测试 ----
        img_tensor = (torch.from_numpy(np.array(attacked_img)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
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
        
        rs_dec = RSCodec(ECC_LEN)
        decoded, _, _ = rs_dec.decode(ext_bytes)
        extracted_text = decoded.strip(b'\0').decode('utf-8', errors='ignore')
        
        extract_success = (extracted_text == secret_text)
        
        print(f"    -> PSNR: {psnr_val:.2f} | JPEG 80 提取: {'✅成功' if extract_success else '❌失败'}")
        
    except Exception as e:
        print(f"    -> ❌ 处理崩溃: {e}")
        psnr_val, ssim_val, extract_success = 0.0, 0.0, False

    results.append({
        "Image_ID": idx,
        "Prompt": prompt[:50],
        "PSNR": round(psnr_val, 2),
        "SSIM": round(ssim_val, 4),
        "JPEG_80_Success": extract_success
    })
    
    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    
    # 释放显卡内存 (绝杀显存溢出)
    gc.collect()
    torch.cuda.empty_cache()

end_time = time.time()
print(f"\n========== 🎉 全部 500 个样本跑测结束！ ==========")
if 'df' in locals():
    print(f"总成功率 (含JPEG 80攻击): {(df['JPEG_80_Success'].sum() / len(df)) * 100:.2f}%")
print(f"总耗时: {(end_time - start_time) / 3600:.2f} 小时")
print(f"最终结果已安全保存在: {CSV_PATH}")