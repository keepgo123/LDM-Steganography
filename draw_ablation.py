import matplotlib.pyplot as plt

# 1. 严格对齐你图片中的实验数据
# 横坐标：Margin Alpha (从 0.0 到 0.2)
alpha_values = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

# 蓝色曲线数据：PSNR (根据图片：从10.3起跳，最终到11.4左右)
psnr_values = [10.32, 10.45, 10.61, 10.78, 10.95, 11.12, 11.25, 11.38, 11.45]

# 红色曲线数据：BER (根据图片：从0.4起跳，最终降到0.1左右)
# 注意：右侧纵坐标 0.4 代表 40% 的误码率
ber_values = [0.41, 0.35, 0.28, 0.22, 0.18, 0.15, 0.13, 0.11, 0.10]

fig, ax1 = plt.subplots(figsize=(8, 5))

# 绘制 PSNR 曲线 (左轴)
color = 'tab:blue'
ax1.set_xlabel('Margin Alpha (α)', fontsize=12)
ax1.set_ylabel('PSNR (dB)', color=color, fontsize=12)
ax1.plot(alpha_values, psnr_values, color=color, marker='o', linewidth=2, label='PSNR')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(10, 12) # 根据图片刻度设置范围

# 实例化第二个坐标轴绘制 BER
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Bit Error Rate (BER)', color=color, fontsize=12)
ax2.plot(alpha_values, ber_values, color=color, marker='s', linestyle='--', linewidth=2, label='BER')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 0.5) # 根据图片刻度设置范围

plt.title('Ablation Study: Margin (α) vs. Quality & Reliability', fontsize=14)
fig.tight_layout()
plt.grid(alpha=0.3)
plt.show()