import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache"

import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
import json

print("========== 毕业设计：补充鲁棒性攻击测试 (高斯噪声 & 随机裁剪) ==========")

# --- 1. 参数设置 ---
STEPS = 50
KEY = 12345
GUIDANCE_SCALE = 3.0  
MSG_LEN = 60   
ECC_LEN = 140  
TOTAL_BYTES = MSG_LEN + ECC_LEN
TEST_NUM = 50 # 抽样 50 张进行极端攻击测试

OUTPUT_DIR = "thesis_final_results"
CSV_PATH = os.path.join(OUTPUT_DIR, "coco_500_results.csv")
STEGO_DIR = os.path.join(OUTPUT_DIR, "stego_images")

# 读取之前跑出来的前50个 Prompt 和 Secret
df_existing = pd.read_csv(CSV_PATH).head(TEST_NUM)

# --- 2. 加载模型 ---
print("正在加载 Stable Diffusion 模型...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", safety_checker=None).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# --- 3. 定义攻击函数 ---
def add_gaussian_noise(img, std=0.05):
    """添加高斯噪声 (模拟信号干扰)"""
    img_arr = np.array(img) / 255.0
    noise = np.random.normal(0, std, img_arr.shape)
    noisy_img_arr = np.clip(img_arr + noise, 0, 1)
    return Image.fromarray((noisy_img_arr * 255).astype(np.uint8))

def center_crop_and_resize(img, crop_ratio=0.9):
    """中心裁剪并缩放回原大小 (模拟截图)"""
    w, h = img.size
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - new_w) / 2, (h - new_h) / 2
    img_cropped = img.crop((left, top, left + new_w, top + new_h))
    return img_cropped.resize((w, h), Image.BILINEAR)

# --- 4. 开始攻击测试 ---
attack_results = []
for idx, row in df_existing.iterrows():
    prompt = row['Prompt']
    secret_text = f"Penghui_Graduation_ID_{idx:04d}_" + os.urandom(8).hex() # 注意：由于随机数种子未保存，这里我们只验证 RS 纠错极限
    # 因为原秘密是随机生成的，我们这里做盲提测试，看提取出来的乱码长度和校验位是否能通过 RS 认证
    
    stego_path = os.path.join(STEGO_DIR, f"stego_{idx:04d}.png")
    if not os.path.exists(stego_path): continue
    
    original_stego = Image.open(stego_path).convert("RGB")
    
    # 制造被攻击的图像
    img_noise = add_gaussian_noise(original_stego, std=0.03)
    img_crop = center_crop_and_resize(original_stego, crop_ratio=0.9)
    
    # 封装提取函数
    def try_extract(attacked_img):
        try:
            img_tensor = (torch.from_numpy(np.array(attacked_img)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
            with torch.no_grad():
                z0 = pipe.vae.encode(img_tensor).latent_dist.mean * pipe.vae.config.scaling_factor
            
            pipe.scheduler.set_timesteps(STEPS, device="cuda")
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            context = torch.cat([pipe.text_encoder(pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda"))[0], pipe.text_encoder(text_input.input_ids.to("cuda"))[0]])
            
            inv_lat = z0.clone()
            for t in pipe.scheduler.timesteps.flip(0):
                with torch.no_grad():
                    noise_pred = pipe.unet(torch.cat([inv_lat] * 2), t, encoder_hidden_states=context).sample
                    noise_pred = noise_pred.chunk(2)[0] + GUIDANCE_SCALE * (noise_pred.chunk(2)[1] - noise_pred.chunk(2)[0])
                alpha_next = pipe.scheduler.alphas_cumprod[t + (1000//STEPS)] if t < 950 else torch.tensor(1.0, device="cuda", dtype=torch.float16)
                inv_lat = alpha_next**0.5 * ((inv_lat - (1 - pipe.scheduler.alphas_cumprod[t])**0.5 * noise_pred) / pipe.scheduler.alphas_cumprod[t]**0.5) + (1 - alpha_next)**0.5 * noise_pred

            flat_inv = inv_lat.view(-1)
            torch.manual_seed(KEY)
            ext_idx = torch.randperm(flat_inv.shape[0])
            ext_bits = ''.join(['1' if flat_inv[ext_idx[j]] > 0 else '0' for j in range(TOTAL_BYTES * 8)])
            ext_bytes = bytearray([int(ext_bits[j:j+8], 2) for j in range(0, len(ext_bits), 8)])
            
            # 只要 RS 校验不抛出异常，说明成功恢复！
            decoded, _, _ = RSCodec(ECC_LEN).decode(ext_bytes)
            return True
        except:
            return False

    print(f">>> 正在测试样本 [{idx+1}/{TEST_NUM}]...")
    success_noise = try_extract(img_noise)
    success_crop = try_extract(img_crop)
    print(f"    高斯噪声: {'✅' if success_noise else '❌'} | 中心裁剪: {'✅' if success_crop else '❌'}")
    
    attack_results.append({"Image_ID": idx, "Noise_Success": success_noise, "Crop_Success": success_crop})

df_attacks = pd.DataFrame(attack_results)
print(f"\n✅ 补充实验完成！")
print(f"高斯噪声 (std=0.03) 提取成功率: {df_attacks['Noise_Success'].mean()*100:.1f}%")
print(f"中心裁剪 (保持90%) 提取成功率: {df_attacks['Crop_Success'].mean()*100:.1f}%")