import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("========== 毕业设计：生成学术级定量评估折线图 ==========")

# --- 1. 设置学术绘图风格与中文字体 ---
# 注意：如果是 Windows 系统，默认使用 SimHei；Mac 系统请改为 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    
sns.set_theme(style="whitegrid", font="SimHei")

# --- 2. 加载你昨晚跑出来的 500 个数据 ---
csv_path = "thesis_final_results/coco_500_results.csv"
if not os.path.exists(csv_path):
    print(f"❌ 找不到数据文件 {csv_path}！请检查路径。")
    exit()

df = pd.read_csv(csv_path)
x = df['Image_ID']

# --- 3. 创建画布 (双 Y 轴折线图) ---
# dpi=300 是学术论文打印的黄金标准分辨率
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300) 

# --- 4. 绘制左轴：PSNR 曲线 ---
# 计算 20 个样本的滑动平均值 (平滑趋势线)
window_size = 20
psnr_smooth = df['PSNR'].rolling(window=window_size, min_periods=1).mean()

color1 = '#1f77b4' # 经典学术蓝
ax1.set_xlabel('图像样本编号 (Image ID / COCO 2017)', fontsize=12, fontweight='bold')
ax1.set_ylabel('峰值信噪比 PSNR (dB)', color=color1, fontsize=12, fontweight='bold')

# 画原始数据的半透明细线 (展现真实波动)
ax1.plot(x, df['PSNR'], color=color1, alpha=0.25, linewidth=1, label='PSNR 原始分布')
# 画平滑后的粗实线 (展现整体趋势)
ax1.plot(x, psnr_smooth, color=color1, linewidth=3, label=f'PSNR 趋势线 (MA={window_size})')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(False) # 关闭左轴内部网格，防止画面太乱

# --- 5. 绘制右轴：SSIM 曲线 ---
ax2 = ax1.twinx()  
color2 = '#ff7f0e' # 经典学术橙
ssim_smooth = df['SSIM'].rolling(window=window_size, min_periods=1).mean()

ax2.set_ylabel('结构相似性 SSIM', color=color2, fontsize=12, fontweight='bold')
# 画原始数据的半透明细线
ax2.plot(x, df['SSIM'], color=color2, alpha=0.25, linewidth=1, label='SSIM 原始分布')
# 画平滑后的粗虚线
ax2.plot(x, ssim_smooth, color=color2, linewidth=3, linestyle='--', label=f'SSIM 趋势线 (MA={window_size})')
ax2.tick_params(axis='y', labelcolor=color2)

# --- 6. 整合图例与标题 ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# 将图例放在图表正上方，显得大气
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=10)

plt.title('图4-2 500张COCO样本的含密图像质量(PSNR/SSIM)分布与趋势', fontsize=14, pad=40, fontweight='bold')
plt.tight_layout()

# --- 7. 保存高斯图 ---
save_path = "thesis_final_results/Figure_4_2_Quality_Chart.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 学术折线图绘制成功！已保存至: {save_path}")
print("快打开图片看看效果吧！")