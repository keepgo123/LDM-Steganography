import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：第四章 嵌入容量极限与抗压测试 ==========")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
STEPS = 50
KEY = 12345

# 我们准备了 4 个级别的秘密信息来测试容量极限
# 注意：RS纠错码的单块最大限制是 255 字节（包含数据+冗余），所以我们测试的字符串要在这个范围内
SECRETS = {
    "Level_1_ID": "2022101196",  # 10 bytes
    "Level_2_Sentence": "Hello World! Stable Diffusion Steganography.", # 44 bytes
    "Level_3_Quote": "Any sufficiently advanced technology is indistinguishable from magic. - Arthur C. Clarke", # 86 bytes
    "Level_4_Poem": "To see a World in a Grain of Sand, And a Heaven in a Wild Flower, Hold Infinity in the palm of your hand, And Eternity in an hour. - William Blake (Auguries of Innocence)" # 174 bytes
}

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompt = "A majestic lion resting in the savannah, highly detailed, 4k"
seed = 2048 
results = []

for level_name, secret_text in SECRETS.items():
    print(f"\n>>> 正在测试 {level_name} | 长度: {len(secret_text)} 字节...")
    
    # 根据信息长度动态调整 RS 冗余度 (保证总长度不超过 255)
    # 给短信息多一点冗余，给长信息少一点冗余以防超出 255 限制
   # 替换为这行（将 32 提高到 120，大幅放开纠错上限）：
    redundancy = min(120, 250 - len(secret_text))
    rs = RSCodec(redundancy)
    encoded_msg = rs.encode(secret_text.encode('utf-8'))
    num_bits = len(encoded_msg) * 8
    
    # 1. 生成含密图
    gen = torch.Generator("cuda").manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=gen, device="cuda", dtype=torch.float16)
    
    flat_lat = latents.clone().view(-1)
    torch.manual_seed(KEY)
    indices = torch.randperm(flat_lat.shape[0])
    binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
    
    for j, bit in enumerate(binary_secret):
        idx = indices[j]
        flat_lat[idx] = 0.8 if bit == '1' else -0.8
        
    stego_img = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), guidance_scale=1.0).images[0]
    
    # 2. 提取逻辑
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

    # 比特提取
    flat_inv = inv_lat.view(-1)
    torch.manual_seed(KEY)
    extract_indices = torch.randperm(flat_inv.shape[0])
    ext_bits = ''.join(['1' if flat_inv[extract_indices[j]] > 0 else '0' for j in range(num_bits)])
    ext_bytes = bytearray([int(ext_bits[j:j+8], 2) for j in range(0, len(ext_bits), 8)])
    
    # 计算误码
    errors = sum([1 for j in range(len(encoded_msg)) if encoded_msg[j] != ext_bytes[j]])
    error_rate = (errors / len(encoded_msg)) * 100
    
    status = "Failed"
    try:
        decoded, _, _ = rs.decode(ext_bytes)
        if decoded.decode('utf-8') == secret_text: 
            status = "Success"
    except: 
        pass
    
    print(f"    原始载荷: {len(secret_text)} B | 编码后载荷: {len(encoded_msg)} B")
    print(f"    产生误码: {errors} 字节 | 误码率: {error_rate:.2f}% | 最终提取: {status}")
    
    results.append({
        "Level": level_name, 
        "Payload(Bytes)": len(secret_text), 
        "Errors": errors, 
        "Error_Rate(%)": round(error_rate, 2),
        "Status": status
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
df.to_csv("thesis_quantitative_results/capacity_limits.csv", index=False)
print("\n🏆 容量极限测试完成！数据已保存至 CSV。")