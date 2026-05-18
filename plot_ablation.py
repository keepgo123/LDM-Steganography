import pandas as pd
import matplotlib.pyplot as plt

print("========== 毕业设计：生成参数消融实验双轴折线图 ==========")

# 1. 尝试读取刚才跑出的 CSV 数据
csv_path = "thesis_quantitative_results/ablation_margin.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"成功读取数据：共 {len(df)} 组参数测试结果。")
except FileNotFoundError:
    print(f"找不到 {csv_path}，请确认上一步消融实验代码是否成功运行并生成了文件。")
    exit()

# 2. 设置学术论文绘图风格
# 设置中文字体，防止中文显示为方块 (Windows 默认 SimHei，Mac 可改成 Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布，设置大小和高 DPI 保证图片清晰度
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

x = df["Margin"]

# 3. 绘制左侧主轴：PSNR (图像质量) - 蓝色实线圆点
color_psnr = '#1f77b4'  # 学术经典蓝
line1 = ax1.plot(x, df["PSNR (dB)"], color=color_psnr, marker='o', linestyle='-', 
                 linewidth=2.5, markersize=8, label='图像质量 PSNR (dB)')
ax1.set_xlabel('嵌入强度参数 (Margin)', fontsize=14, fontweight='bold')
ax1.set_ylabel('图像质量 PSNR (dB)', color=color_psnr, fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_psnr, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.6) # 添加淡色网格线方便读数

# 4. 绘制右侧次轴：BER (误码率) - 红色实线方块
ax2 = ax1.twinx()  # 共享 X 轴
color_ber = '#d62728'   # 学术经典红
line2 = ax2.plot(x, df["BER (%)"], color=color_ber, marker='s', linestyle='-', 
                 linewidth=2.5, markersize=8, label='底层误码率 BER (%)')
ax2.set_ylabel('底层误码率 BER (%)', color=color_ber, fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_ber, labelsize=12)

# 5. 合并并设置图例 (Legend)
lines = line1 + line2
labels = [l.get_label() for l in lines]
# 将图例放在正上方，避免遮挡数据线
ax1.legend(lines, labels, loc='upper center', fontsize=12, framealpha=0.9, edgecolor='gray')

# 6. 设置大标题并调整布局
plt.title('参数消融实验：不可感知性与鲁棒性的折中关系', fontsize=16, fontweight='bold', pad=15)
fig.tight_layout()  # 自动调整边距，防止标签被截断

# 7. 保存为高清图片
save_path = "thesis_quantitative_results/ablation_tradeoff_curve.png"
plt.savefig(save_path, format='png', bbox_inches='tight')
print(f"🏆 图表已成功生成并保存为超高清图片：{save_path}")

# 弹出窗口预览
plt.show()