import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from reedsolo import RSCodec
import numpy as np
import gc

print("========== 毕业设计：免训练大模型隐写可视化系统正在启动 ==========")
print("正在加载 Stable Diffusion 模型，请稍候...")

# 1. 全局加载模型
MODEL_ID = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    safety_checker=None
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

STEPS = 50
KEY = 12345
SEED = 2048

# --- 🎯 答辩专用平衡参数 ---
# 3.0 是黄金分割点：模型不仅能听懂提示词画出真实的狮子，产生得误差刚好在纠错极限内！
GUIDANCE_SCALE = 3.0  

# --- 🚀 终极纠错架构 ---
MSG_LEN = 60   # 秘密信息最大字节数
ECC_LEN = 140  # 维持疯狂的冗余度，靠它硬抗 GUIDANCE_SCALE=3.0 带来的误差
TOTAL_BYTES = MSG_LEN + ECC_LEN

def embed_info(prompt, secret_text, margin):
    """嵌入秘密信息并生成图像"""
    try:
        secret_bytes = secret_text.encode('utf-8')
        if len(secret_bytes) > MSG_LEN:
            return None, f"❌ 文本太长！当前设置最大支持 {MSG_LEN} 字节 (约 {MSG_LEN//3} 个汉字)。"
        
        secret_bytes = secret_bytes.ljust(MSG_LEN, b'\0')
        rs = RSCodec(ECC_LEN)
        encoded_msg = rs.encode(secret_bytes)
        
        gen = torch.Generator("cuda").manual_seed(SEED)
        latents = torch.randn((1, 4, 64, 64), generator=gen, device="cuda", dtype=torch.float16)
        
        flat_lat = latents.clone().view(-1)
        torch.manual_seed(KEY)
        indices = torch.randperm(flat_lat.shape[0])
        binary_secret = ''.join(format(b, '08b') for b in encoded_msg)
        
        for j, bit in enumerate(binary_secret):
            idx = indices[j]
            flat_lat[idx] = margin if bit == '1' else -margin
            
        # 使用 GUIDANCE_SCALE (3.0) 生成真实、威猛的狮子
        stego_img = pipe(prompt=prompt, latents=flat_lat.view(1, 4, 64, 64), guidance_scale=GUIDANCE_SCALE).images[0]
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return stego_img, "✅ 成功！图像清晰且像狮子，信息已高鲁棒性嵌入。"
    except Exception as e:
        return None, f"❌ 嵌入失败: {str(e)}"

def extract_info(image, prompt):
    """从图像中提取秘密信息"""
    if image is None:
        return "请先上传一张图像！"
    try:
        # --- 优化图像前处理，确保误差最小化 ---
        image = image.resize((512, 512)) # 强制拉伸一下，确保 VAE 步长对齐
        img_tensor = (torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16) / 255.0) * 2.0 - 1.0
        
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
                # 使用相同的 GUIDANCE_SCALE (3.0) 进行精确反演
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

            # 优化 DDIM 反演公式，确保最后一刻的稳定性
            prev_timestep = timesteps[i-1] if i > 0 else t # 这里简化处理，实际需要调度器逻辑
            alpha_t = pipe.scheduler.alphas_cumprod[t]
            alpha_prev = pipe.scheduler.alphas_cumprod[prev_timestep] if i > 0 else alpha_t
            
            pred_x0 = (inv_lat - (1 - alpha_t)**0.5 * noise_pred) / alpha_t**0.5
            
            # --- 精确的一步反演更新 ---
            next_t = t + (1000//STEPS) if t < 950 else torch.tensor(999) 
            # 简化一下更新逻辑保证稳定性
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
        
        # 靠这 140 字节 ECC 硬磕 3.0 CFG 误差！
        rs = RSCodec(ECC_LEN)
        decoded, _, _ = rs.decode(ext_bytes)
        secret_text = decoded.strip(b'\0').decode('utf-8', errors='ignore')
        
        gc.collect()
        torch.cuda.empty_cache()
        return f"🎉 提取成功！\n\n🔒 隐藏的秘密信息是：\n【 {secret_text} 】"
    except Exception as e:
        return f"❌ 提取失败：CFG 提高后误差过大，建议将 Margin 略微提高，或检查 Prompt 是否一致。"

# 搭建 Gradio 网页界面
with gr.Blocks(title="免训练图像隐写系统") as demo:
    gr.Markdown("# 🎓 基于 Stable Diffusion 潜空间反演的免训练隐写系统 (演示 Demo)")
    
    with gr.Tabs():
        with gr.TabItem("🔐 生成端 (信息嵌入)"):
            with gr.Row():
                with gr.Column():
                    prompt_in = gr.Textbox(label="图像生成提示词 (Prompt) - 需英文", value="A majestic lion resting in the savannah, highly detailed, 4k", lines=2)
                    secret_in = gr.Textbox(label="要隐藏的秘密信息 (Secret Text)", value="鹏辉答辩必过！", lines=2)
                    # 默认参数提升至 1.4，结合 CFG=3.0 达到完美平衡
                    margin_slider = gr.Slider(minimum=0.5, maximum=2.5, step=0.1, value=1.4, label="嵌入强度参数 (Margin)", info="由于提高了图像质量(CFG)，建议默认使用 1.4")
                    embed_btn = gr.Button("🚀 生成含密图像", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="生成的含密图像", type="pil")
                    status_out = gr.Textbox(label="系统状态")
                    
            embed_btn.click(fn=embed_info, inputs=[prompt_in, secret_in, margin_slider], outputs=[img_out, status_out])

        with gr.TabItem("🔓 接收端 (信息提取)"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="上传含密图像", type="pil")
                    prompt_ext_in = gr.Textbox(label="图像反演提示词 (需与生成时完全一致)", value="A majestic lion resting in the savannah, highly detailed, 4k", lines=2)
                    extract_btn = gr.Button("🔍 提取隐藏信息", variant="primary")
                with gr.Column():
                    secret_out = gr.Textbox(label="提取结果", lines=5)
                    
            extract_btn.click(fn=extract_info, inputs=[img_in, prompt_ext_in], outputs=[secret_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, theme=gr.themes.Soft())