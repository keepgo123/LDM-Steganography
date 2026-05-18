import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

# ==========================================
# 1. 字体与画布设置
# ==========================================
# 确保中文字体正常显示，优先使用微软雅黑，备用黑体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

# 【核心参数】超大画布 (20x16) + 极限高清 DPI (600)
fig, ax = plt.subplots(figsize=(20, 16), dpi=600)
ax.set_axis_off()
ax.set_xlim(0, 20)
ax.set_ylim(-1.5, 15.5)

# ==========================================
# 2. 绘图辅助函数（动态自适应文本框，杜绝越界）
# ==========================================
def draw_node(x, y, text, color, fs=15, text_color='black'):
    """使用 Matplotlib 的 bbox 属性自动生成完美包裹的圆角文本框"""
    bbox_props = dict(boxstyle="round,pad=0.7", fc=color, ec='#444444', lw=2)
    # zorder=5 确保文本框在最上层，压住箭头线
    ax.text(x, y, text, ha="center", va="center", size=fs, weight='bold', 
            color=text_color, bbox=bbox_props, zorder=5)

def draw_arrow(x1, y1, x2, y2, rad=0.0, style="-|>", lw=2.5, ls='-', color='#555555'):
    """绘制底层连线，zorder=1 确保线藏在文本框下面，极其整洁"""
    arrowprops = dict(arrowstyle=style, color=color, lw=lw, ls=ls, 
                      connectionstyle=f"arc3,rad={rad}", mutation_scale=20)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops, zorder=1)

def draw_bg(y_center, title, color):
    """绘制四个阶段的底色带与标题"""
    bg = patches.Rectangle((0.5, y_center-1.9), 19, 3.8, facecolor=color, 
                           edgecolor='none', alpha=0.2, zorder=0)
    ax.add_patch(bg)
    ax.text(1.0, y_center + 1.2, title, fontsize=18, weight='bold', color='#333333', zorder=2)

# ==========================================
# 3. 绘制四大阶段背景区块
# ==========================================
draw_bg(13, "阶段一：秘密文本预处理", '#90caf9') # 浅蓝
draw_bg(9,  "阶段二：嵌入位置生成",   '#a5d6a7') # 浅绿
draw_bg(5,  "阶段三：噪声调制阶段",   '#ffcc80') # 浅橙
draw_bg(1,  "阶段四：图像生成阶段",   '#ce93d8') # 浅紫

# ==========================================
# 4. 绘制核心模块 (节点)
# ==========================================
# ---- 阶段 1 ----
draw_node(2, 13, "待隐藏短文本\n(如单词 'cat')", '#ffffff')
draw_node(7, 13, "转 UTF-8 并填充\n(空字节补齐至 10 Bytes)", '#ffffff')
draw_node(12, 13, "Reed-Solomon 编码器\n(+20 Bytes ECC, 总长30)", '#ffffff')
draw_node(17, 13, "展开为待嵌入\n机密比特串\n(total_bits = 240)", '#ffeb3b') # 黄色高亮

# ---- 阶段 2 ----
draw_node(2, 9, "私钥图像 (private_cat.jpg)\n与 私钥文本 ('mykey')", '#ffffff')
draw_node(7, 9, "分别计算 SHA-256\n拼接后再次哈希", '#ffffff')
draw_node(12, 9, "生成 256 位\n确定性随机种子", '#ffffff')
draw_node(17, 9, "伪随机序列发生器\n获取 240 个嵌入位置\n(从 16384 全排列中提取)", '#ffeb3b') # 黄色高亮

# ---- 阶段 3 ----
draw_node(7, 5, "标准高斯噪声张量\n(1 × 4 × 64 × 64 展平)", '#ffffff')
draw_node(12, 5, "核心噪声调制逻辑\n\n当前比特 = 1 ➜ 置为 +δ (2.8)\n当前比特 = 0 ➜ 置为 -δ (-2.8)\n(其余位置保持不变)", '#ffffff')
draw_node(17, 5, "调制后的初始噪声\n(含密潜变量)", '#ffeb3b') # 黄色高亮

# ---- 阶段 4 ----
draw_node(7, 1, "公钥文本提示词\n('A majestic mountain...')", '#ffffff')
draw_node(12, 1, "Stable Diffusion 流水线 (txt2img)\n\nDDIM 采样步数 = 50\n引导尺度 CFG = 7.5\n确定性采样 (η = 0)", '#ffffff')
draw_node(17, 1, "最终含密图像\n(512 × 512 RGB)\n视觉与自然图像无异", '#a5d6a7', fs=16) # 最终结果绿色高亮

# ==========================================
# 5. 绘制逻辑连线
# ==========================================
# 阶段内的水平推进流
draw_arrow(2, 13, 7, 13); draw_arrow(7, 13, 12, 13); draw_arrow(12, 13, 17, 13)
draw_arrow(2, 9, 7, 9);   draw_arrow(7, 9, 12, 9);   draw_arrow(12, 9, 17, 9)
draw_arrow(7, 5, 12, 5);  draw_arrow(12, 5, 17, 5)
draw_arrow(7, 1, 12, 1);  draw_arrow(12, 1, 17, 1)

# 跨阶段的数据馈送流 (带有红色虚线与标签)
bbox_red = dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d32f2f", lw=1.5)

# 比特流 -> 调制
draw_arrow(17, 13, 12, 5, rad=0.2, ls='--', color='#d32f2f')
ax.text(15.2, 9.3, "注入 240 位比特", ha="center", va="center", size=13, weight='bold', 
        color='#d32f2f', bbox=bbox_red, zorder=6)

# 位置流 -> 调制
draw_arrow(17, 9, 12, 5, rad=-0.2, ls='--', color='#d32f2f')
ax.text(14.8, 7.0, "提供指定坐标", ha="center", va="center", size=13, weight='bold', 
        color='#d32f2f', bbox=bbox_red, zorder=6)

# 潜变量 -> SD 模型
draw_arrow(17, 5, 12, 1, rad=-0.2, ls='--', color='#d32f2f')
ax.text(14.8, 3.0, "作为初始潜变量", ha="center", va="center", size=13, weight='bold', 
        color='#d32f2f', bbox=bbox_red, zorder=6)

# ==========================================
# 6. 保存超清双重格式
# ==========================================
plt.tight_layout()

# 格式1：SVG矢量图（论文首选！插入Word无限放大绝对清晰）
plt.savefig("stego_4stage_flowchart_ULTRA.svg", format='svg', bbox_inches='tight')

# 格式2：600 DPI 极限清PNG
plt.savefig("stego_4stage_flowchart_ULTRA.png", dpi=600, bbox_inches='tight')

print("✅ 成功生成科研级无损图表！")
print("👉 请在你的文件夹中找到 'stego_4stage_flowchart_ULTRA.svg'")
print("👉 直接将 .svg 文件拖入 Word 中，彻底告别模糊！")
plt.show()