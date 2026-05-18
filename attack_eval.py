import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
import numpy as np
import pandas as pd
import os
import gc
from PIL import Image
import io

print("========== 毕业设计精进实验：图像抗攻击与鲁棒性评估 ==========")

# --- 1. 初始化设置 ---
STEPS = 50
KEY = 12345
GUIDANCE_SCALE = 3.0  
MSG_LEN = 60   
ECC_LEN = 140  
TOTAL_BYTES = MSG_LEN + ECC_LEN

OUTPUT_DIR = "thesis_results_batch"
STEGO_DIR = os.path.join(OUTPUT_DIR, "stego_images")

print("正在加载 Stable Diffusion 模型 (FP16 加速模式)...")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    variant="fp16",
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 刚才测试的 5 个 Prompt 和 Secret (需保持完全一致)
test_prompts = [
    "A majestic lion resting in the savannah, highly detailed, 4k",
    "A vintage red sports car parked on a city street at night",
    "A cute golden retriever playing with a frisbee in the park",
    "A cozy living room with a fireplace and modern furniture",
    "A delicious pepperoni pizza on a wooden table, food photography"
]
secret_messages = [
    "Test Message 1: Hello World!",
    "Test Message 2: Stealth is key.",
    "Test Message 3: Diffusion models rock.",
    "Test Message 4: Security verified.",
    "Test Message 5: Graduation defense prep."
]

# 测试的 JPEG 压缩质量梯度 (100为无损，数值越小破坏越大)
jpeg_qualities = [90, 80, 70]
robustness_results = []

print("\n🚀 开始 JPEG 压缩攻击测试...")

for idx, (prompt, secret_text) in enumerate(zip(test_prompts, secret_messages)):
    stego_path = os.path.join(STEGO_DIR, f"stego_{idx}.png")
    
    if not os.path.exists(stego_path):
        print(f"找不到图像 {stego_path}，请确保上一个脚本跑通了。")
        continue
        
    original_stego_img = Image.open(stego_path).convert("RGB")
    print(f"\n>>> 正在攻击测试样本 [{idx+1}/5]: {prompt[:20]}...")
    
    for quality in jpeg_qualities:
        # 1. 模拟信道攻击：JPEG 压缩
        img_byte_arr = io.BytesIO()
        original_stego_img.save(img_byte_arr, format='JPEG', quality=quality)
        attacked_img = Image.open(img_byte_arr).convert("RGB")
        
        # 2. 从被攻击（变模糊）的图像中尝试提取信息
        try:
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
            
            rs = RSCodec(ECC_LEN)
            decoded, _, _ = rs.decode(ext_bytes)
            extracted_text = decoded.strip(b'\0').decode('utf-8', errors='ignore')
            
            extract_success = (extracted_text == secret_text)
            print(f"    JPEG 质量 {quality} -> 提取{'成功 ✅' if extract_success else '失败 ❌'}")
            
        except Exception as e:
            extract_success = False
            print(f"    JPEG 质量 {quality} -> 提取失败 ❌ (超出纠错极限)")
            
        robustness_results.append({
            "Image_ID": idx,
            "JPEG_Quality": quality,
            "Extraction_Success": extract_success
        })
        
        gc.collect()
        torch.cuda.empty_cache()

# --- 3. 汇总数据 ---
df_robust = pd.DataFrame(robustness_results)
csv_path = os.path.join(OUTPUT_DIR, "robustness_results.csv")
df_robust.to_csv(csv_path, index=False)

print("\n========== 攻击测试完成！==========")
for q in jpeg_qualities:
    subset = df_robust[df_robust['JPEG_Quality'] == q]
    success_rate = (subset['Extraction_Success'].sum() / len(subset)) * 100
    print(f"JPEG 压缩质量 {q} 下的提取成功率: {success_rate:.2f}%")
print(f"数据已保存至 {csv_path}。拿着这个数据，你的导师绝对挑不出毛病！")