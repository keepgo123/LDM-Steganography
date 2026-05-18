import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
from PIL import Image
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import time
import warnings
warnings.filterwarnings("ignore")

print("========== 毕业设计：第一阶段 (50 图自动化定量实验) ==========")

SECRET = "2022101196"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# 🌟 我们的黄金基准参数 (主攻稳定性)
MARGIN = 0.65
RS_REDUNDANCY = 16   # 容错 8 字节
STEPS = 50      
KEY = 12345

# 🎨 50个高多样性测试集 (涵盖风景、建筑、动物、科幻、物体)
TEST_PROMPTS = [
    # 风景与自然
    "A stunning sunset over the Great Wall of China, cinematic lighting",
    "A dense tropical rainforest with sunlight filtering through leaves",
    "Snow-capped mountains reflecting in a crystal clear lake",
    "A vast desert with rolling sand dunes under a starry night sky",
    "A dramatic waterfall cascading down a rocky cliff in autumn",
    "A peaceful meadow filled with blooming wildflowers at dawn",
    "Aurora borealis dancing over a frozen tundra landscape",
    "A misty morning in a dense bamboo forest",
    "Volcanic eruption with glowing red lava flows at night",
    "A serene beach with white sand and turquoise water",
    # 城市与建筑
    "Minimalist modern architecture skyscraper against a clear blue sky",
    "A bustling cyberpunk city street with neon signs in the rain",
    "A cozy wooden cabin covered in snow in a pine forest",
    "An ancient Gothic cathedral with intricate stained glass windows",
    "A futuristic floating city above the clouds",
    "Traditional Japanese temple surrounded by cherry blossoms",
    "A high-speed train zooming past a modern metropolis",
    "A quiet European cobblestone alleyway with cafes",
    "An abandoned industrial factory reclaimed by nature",
    "A majestic medieval castle on a high cliff overlooking the sea",
    # 动物与生物
    "A cute cat sitting on a windowsill in an oil painting style",
    "A fierce tiger prowling through the jungle underbrush",
    "A majestic eagle soaring high above the clouds",
    "A colorful parrot perched on a tropical branch",
    "A pack of wolves howling at the full moon",
    "A gentle elephant walking through the savanna",
    "A glowing deep-sea jellyfish swimming in the dark ocean",
    "A tiny hummingbird hovering near a bright red flower",
    "A sleeping red panda on a tree branch",
    "A school of vibrant tropical fish in a coral reef",
    # 科幻与奇幻
    "A futuristic laboratory with glowing blue lights and robots",
    "An astronaut floating in space with Earth in the background",
    "A giant spaceship landing on an alien planet with red sand",
    "A magical glowing tree in the center of a dark enchanted forest",
    "A steampunk airship flying through a cloudy sky",
    "A cyborg samurai holding a glowing katana",
    "A portal opening to another dimension filled with swirling colors",
    "A massive mechanical dragon breathing blue fire",
    "A bustling market on a space station",
    "A glowing magic potion bottle sitting on an old wooden desk",
    # 物体与静物
    "A vintage typewriter with a blank piece of paper on a desk",
    "A cup of steaming hot coffee with latte art on a wooden table",
    "An intricate pocket watch with exposed golden gears",
    "A classic acoustic guitar resting against a brick wall",
    "A beautiful bouquet of red roses in a glass vase",
    "A glowing lightbulb suspended in the dark",
    "A detailed macro shot of a single water drop on a green leaf",
    "An old leather-bound book with glowing magical runes",
    "A shiny grand piano in an empty concert hall",
    "A colorful hot air balloon flying in a clear blue sky"
]

os.makedirs("thesis_quantitative_results", exist_ok=True)
rs = RSCodec(RS_REDUNDANCY)
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

encoded_msg = rs.encode(SECRET.encode('utf-8'))
binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
num_bits = len(binary_secret)

print(f"📊 系统参数：测试集 {len(TEST_PROMPTS)} 张 | 冗余 {RS_REDUNDANCY} 字节 | 嵌入点数：{num_bits}\n")

results = []
start_time = time.time()

for i, prompt in enumerate(TEST_PROMPTS):
    try:
        print(f"[{i+1}/{len(TEST_PROMPTS)}] 正在处理: {prompt[:35]}...")
        seed = 1000 + i
        
        # 1. 纯净原图
        gen_ref = torch.Generator("cuda").manual_seed(seed)
        lat_pure = torch.randn((1, 4, 64, 64), generator=gen_ref, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            img_ref = pipe(prompt=prompt, latents=lat_pure, num_inference_steps=STEPS, guidance_scale=1.0).images[0]
        
        # 2. 潜空间嵌入
        flat_lat = lat_pure.clone().view(-1)
        torch.manual_seed(KEY)
        indices = torch.randperm(flat_lat.shape[0])
        for j, bit in enumerate(binary_secret):
            idx = indices[j]
            orig = flat_lat[idx].item()
            if bit == '1':
                flat_lat[idx] = max(orig, MARGIN) if orig > 0 else MARGIN
            else:
                flat_lat[idx] = min(orig, -MARGIN) if orig < 0 else -MARGIN
                
        # 3. 含密图生成
        with torch.no_grad():
            img_stego = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), num_inference_steps=STEPS, guidance_scale=1.0).images[0]
        
        # 4. 指标计算
        ref_np = np.array(img_ref)
        stego_np = np.array(img_stego)
        cur_psnr = psnr(ref_np, stego_np)
        cur_ssim = ssim(ref_np, stego_np, channel_axis=2)
        
        # 只保存前5张作为图表示例，省硬盘空间
        if i < 5:
            img_stego.save(f"thesis_quantitative_results/stego_sample_{i+1}.png")
        
        # 5. 逆向提取
        img_tensor = (torch.from_numpy(stego_np).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
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

        flat_inv = inverted_lat.view(-1)
        torch.manual_seed(KEY)
        extract_indices = torch.randperm(flat_inv.shape[0])
        extracted_bits = ''.join(['1' if flat_inv[extract_indices[j]] > 0 else '0' for j in range(num_bits)])
        extracted_bytes = bytearray([int(extracted_bits[j:j+8], 2) for j in range(0, len(extracted_bits), 8)])
        
        errors_before = sum([1 for j in range(len(encoded_msg)) if encoded_msg[j] != extracted_bytes[j]])
        
        status = "Failed"
        try:
            decoded_msg, _, _ = rs.decode(extracted_bytes)
            if decoded_msg.decode('utf-8') == SECRET:
                status = "Success"
                print(f"    ✅ 成功 | PSNR: {cur_psnr:.2f}dB | 错误: {errors_before}字节")
        except:
            print(f"    ❌ 失败 | PSNR: {cur_psnr:.2f}dB | 错误: {errors_before}字节")

        results.append({
            "ID": i + 1,
            "Category": ["风景", "建筑", "动物", "科幻", "物体"][i // 10],
            "PSNR": round(cur_psnr, 2),
            "SSIM": round(cur_ssim, 4),
            "Errors_Before_RS": errors_before,
            "Extraction": status
        })
        
    except Exception as e:
        print(f"    ⚠️ 发生异常: {e}")
        continue

# 统计并保存最终大表
df = pd.DataFrame(results)
df.to_csv("thesis_quantitative_results/Full_50_Dataset_Report.csv", index=False)

success_rate = (df['Extraction'] == 'Success').mean() * 100
avg_psnr = df['PSNR'].mean()
avg_ssim = df['SSIM'].mean()

print("\n" + "="*50)
print("🎉 测试跑完啦！论文核心数据出炉：")
print(f"总耗时: {(time.time() - start_time)/60:.1f} 分钟")
print(f"平均 PSNR: {avg_psnr:.2f} dB")
print(f"平均 SSIM: {avg_ssim:.4f}")
print(f"综合提取成功率: {success_rate:.1f}%")
print("完整数据已保存至 thesis_quantitative_results/Full_50_Dataset_Report.csv")
print("="*50)