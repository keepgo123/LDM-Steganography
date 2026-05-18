import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：基于 RS 纠错码的鲁棒性扩散隐写系统 (变量名修复版) ==========")

# --- 1. 配置区 ---
SECRET = "2022101196" 
MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT = "A beautiful cozy wooden cabin in a snowy mountain landscape, 4k resolution"
SEED = 8888
KEY = 12345
STEPS = 20

# --- 2. RS 编码阶段 ---
rs = RSCodec(16) # 冗余16字节，可修正8个字节
encoded_msg = rs.encode(SECRET.encode('utf-8'))
binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
num_bits = len(binary_secret)
print(f"[1/5] RS编码完成：编码后长度 {len(encoded_msg)} 字节 (可纠错上限: 8字节)")

# --- 3. 发送方：嵌入与生成 ---
print("正在加载扩散模型...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

print("[2/5] 正在注入学号 (力度 0.6)...")
generator = torch.Generator("cuda").manual_seed(SEED)
latents = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)
flat_latents = latents.view(-1)

torch.manual_seed(KEY)
indices = torch.randperm(flat_latents.shape[0])

for i, bit in enumerate(binary_secret):
    idx = indices[i]
    val = max(abs(flat_latents[idx]), 0.6) 
    flat_latents[idx] = val if bit == '1' else -val

latents = flat_latents.view(1, 4, 64, 64)

print("[3/5] 正在生成图像...")
with torch.no_grad():
    image = pipe(prompt=PROMPT, latents=latents, num_inference_steps=STEPS, guidance_scale=1.0).images[0]

image.save("robust_stego_image.png")
print("      => 图片已保存！")

# --- 4. 接收方：逆向提取 ---
print("\n[4/5] 接收方启动时光机 (DDIM Inversion)...")
img_pil = Image.open("robust_stego_image.png").convert("RGB")
img_np = np.array(img_pil).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16)
img_tensor = 2.0 * img_tensor - 1.0

with torch.no_grad():
    z0 = pipe.vae.encode(img_tensor).latent_dist.mean * pipe.vae.config.scaling_factor

pipe.scheduler.set_timesteps(STEPS, device="cuda")
timesteps = pipe.scheduler.timesteps.flip(0)
text_emb = pipe.text_encoder(pipe.tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda"))[0]

inverted_lat = z0.clone()
for t in timesteps:
    alpha_t = pipe.scheduler.alphas_cumprod[t]
    next_t = t + (1000//STEPS) if t < 950 else torch.tensor(999) 
    alpha_next = pipe.scheduler.alphas_cumprod[next_t] if next_t < 1000 else torch.tensor(1.0)
    
    with torch.no_grad():
        noise_pred = pipe.unet(inverted_lat, t, encoder_hidden_states=text_emb).sample
    pred_x0 = (inverted_lat - (1 - alpha_t)**0.5 * noise_pred) / alpha_t**0.5
    inverted_lat = alpha_next**0.5 * pred_x0 + (1 - alpha_next)**0.5 * noise_pred

# --- 5. 解码与纠错 ---
print("[5/5] 正在提取并进行 RS 纠错...")
# 修正后的变量名：inverted_lat
flat_inv = inverted_lat.view(-1) 
torch.manual_seed(KEY)
extract_indices = torch.randperm(flat_inv.shape[0])

extracted_bits = ""
for i in range(num_bits):
    extracted_bits += '1' if flat_inv[extract_indices[i]] > 0 else '0'

extracted_bytes = bytearray()
for i in range(0, len(extracted_bits), 8):
    extracted_bytes.append(int(extracted_bits[i:i+8], 2))

# 统计纠错前的原始错误数
errors_before = 0
for i in range(len(encoded_msg)):
    if encoded_msg[i] != extracted_bytes[i]:
        errors_before += 1
print(f"📊 统计：纠错前检测到 {errors_before} 个错误字节。")

try:
    decoded_msg, decoded_full, err_count = rs.decode(extracted_bytes)
    final_text = decoded_msg.decode('utf-8')
    print(f"\n==================================================")
    print(f"✅ 成功！RS码修正了 {err_count} 个错误字节。")
    print(f"🎉 最终完美还原学号: {final_text}")
    print(f"==================================================\n")
except Exception as e:
    print(f"\n❌ 纠错失败：错误数 ({errors_before}) 超过了上限 (8)。")