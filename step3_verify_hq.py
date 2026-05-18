import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
from PIL import Image
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：终极闭环提取测试 (抗像素微调干扰) ==========")

SECRET = "2022101196"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
STEPS = 50
KEY = 12345
RS_REDUNDANCY = 16

TEST_PROMPTS = [
    "A stunning sunset over the Great Wall of China, cinematic lighting",
    "A futuristic laboratory with glowing blue lights and robots",
    "A cute cat sitting on a windowsill in an oil painting style",
    "A dense tropical rainforest with sunlight filtering through leaves",
    "Minimalist modern architecture skyscraper against a clear blue sky"
]

rs = RSCodec(RS_REDUNDANCY)
encoded_msg = rs.encode(SECRET.encode('utf-8'))
num_bits = len(encoded_msg) * 8

print("正在加载扩散模型 (用于逆向推导)...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

success_count = 0

for i, prompt in enumerate(TEST_PROMPTS):
    img_path = f"experiment_results_final/images/stego_best_{i+1}.png"
    if not os.path.exists(img_path):
        print(f"找不到图片 {img_path}，跳过。")
        continue
        
    print(f"\n>>> 正在验证第 {i+1} 张高保真图片...")
    
    # 1. 读取受过 5% 微调干扰的图片
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16)
    img_tensor = 2.0 * img_tensor - 1.0

    # 2. VAE 编码与 DDIM 逆向推导
    with torch.no_grad():
        z0 = pipe.vae.encode(img_tensor).latent_dist.mean * pipe.vae.config.scaling_factor

    pipe.scheduler.set_timesteps(STEPS, device="cuda")
    text_emb = pipe.text_encoder(pipe.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda"))[0]
    inverted_lat = z0.clone()
    
    for t in pipe.scheduler.timesteps.flip(0):
        alpha_t = pipe.scheduler.alphas_cumprod[t]
        next_t = t + (1000//STEPS) if t < 950 else torch.tensor(999) 
        alpha_next = pipe.scheduler.alphas_cumprod[next_t] if next_t < 1000 else torch.tensor(1.0)
        
        with torch.no_grad():
            noise_pred = pipe.unet(inverted_lat, t, encoder_hidden_states=text_emb).sample
        pred_x0 = (inverted_lat - (1 - alpha_t)**0.5 * noise_pred) / alpha_t**0.5
        inverted_lat = alpha_next**0.5 * pred_x0 + (1 - alpha_next)**0.5 * noise_pred

    # 3. 提取比特并纠错
    flat_inv = inverted_lat.view(-1)
    torch.manual_seed(KEY)
    extract_indices = torch.randperm(flat_inv.shape[0])

    extracted_bits = ''.join(['1' if flat_inv[extract_indices[j]] > 0 else '0' for j in range(num_bits)])
    extracted_bytes = bytearray([int(extracted_bits[j:j+8], 2) for j in range(0, len(extracted_bits), 8)])

    # 统计纠错前的原始错误
    errors_before = sum([1 for j in range(len(encoded_msg)) if encoded_msg[j] != extracted_bytes[j]])
    
    try:
        decoded_msg, _, err_count = rs.decode(extracted_bytes)
        final_text = decoded_msg.decode('utf-8')
        if final_text == SECRET:
            print(f"    ✅ 提取成功！(原始错误: {errors_before} 字节 -> 纠错后恢复学号: {final_text})")
            success_count += 1
        else:
            print(f"    ⚠️ 纠错完成，但内容不匹配。")
    except Exception as e:
        print(f"    ❌ 提取失败：像素干扰导致错误过多 (原始错误: {errors_before} 字节)。")

print(f"\n==================================================")
print(f"🏆 最终成绩：5 张高保真图片，成功提取了 {success_count} 张！")
print(f"==================================================")