# ==============================================
# 超级电容器前驱体分类（按NAME列同名分组）与性能可视化
# 核心逻辑：基于NAME列（前驱体名称）分组→关联目标性能（SC/SE）
# 已拆分SC/SE性能分布图为两个独立文件
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --------------------------
# saxs. 基础设置（字体/路径）
# --------------------------
# 图表字体（英文显示正常）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300  # 关键修改：设置显示DPI

# 文件路径
file_path = 'wqwdata.xlsx'
# 输出路径（拆分SC/SE为两个独立文件）
output_path_sc = 'Precursor_Name-SC_Distribution.png'
output_path_se = 'Precursor_Name-SE_Distribution.png'
output_path_2 = 'Precursor_Name-Overall_Performance.png'
output_path_3 = 'Precursor_Name_Results.csv'

# --------------------------
# 2. 数据加载与前驱体名称字段确认（已设为NAME列）
# --------------------------
print("=== 步骤1：数据加载与前驱体名称字段确认 ===")
# 读取数据（若Excel格式有问题，可尝试添加 engine='openpyxl' 参数）
try:
    df = pd.read_excel(file_path)
except Exception as e:
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"提示：使用openpyxl引擎读取Excel，解决格式兼容问题")

print(f"原始数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n已确认前驱体名称列：NAME（数据中第1列）")

# 统计NAME列的前驱体类别与样本数
precursor_name_col = "NAME"  # 核心：指定前驱体名称列为NAME
precursor_count = df[precursor_name_col].value_counts()
print(f"该列包含的前驱体类别数量：{df[precursor_name_col].nunique()} 种")
print(f"各类别样本数量分布（前10种）：")
for i, (name, count) in enumerate(precursor_count.head(10).items(), 1):
    print(f"  {i:2d}. {name}：{count} 个样本")
if len(precursor_count) > 10:
    print(f"  ... 其余 {len(precursor_count) - 10} 种前驱体（样本数较少）")

# --------------------------
# 3. 数据预处理（保留核心特征）
# --------------------------
print("\n=== 步骤2：数据预处理 ===")
# 核心特征：NAME（前驱体）+ 目标性能（SC/SE）+ 关键输入特征
target_features = ['SC', 'SE']  # 比容量、比能量
input_features = ['TEMP', 'RT', 'SSA', 'CYCLE']  # 热解温度、保温时间等
core_features = [precursor_name_col] + target_features + [f for f in input_features if f in df.columns]

# 处理缺失值（保留核心特征完整的行）
df_clean = df.dropna(subset=core_features)
drop_count = df.shape[0] - df_clean.shape[0]
print(f"清理前数据量：{df.shape[0]} 行 | 清理后数据量：{df_clean.shape[0]} 行")
print(f"删除含缺失值的行数：{drop_count} 行（{drop_count / df.shape[0] * 100:.1f}%）")

# 保存清理后的结果（含原始NAME列和所有特征）
df_clean.to_csv(output_path_3, index=False, encoding='utf-8-sig')
print(f"预处理结果已保存至：{output_path_3}")

# 获取最终参与分析的前驱体列表
precursor_names = df_clean[precursor_name_col].unique()
n_precursors = len(precursor_names)
print(f"\n最终参与分析的前驱体类别：{n_precursors} 种")

# --------------------------
# 4. 可视化1：拆分SC和SE为两个独立图表
# --------------------------
print("\n=== 步骤3：生成可视化图表（性能分布） ===")
# 配色与标记（适配多类别，最多支持20种，超过自动循环）
colors = plt.cm.Set3(np.linspace(0, 1, min(n_precursors, 60)))
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h'][:n_precursors]

# ========== 4.saxs 生成SC（比容量）独立分布图 ==========
fig, ax = plt.subplots(figsize=(9.65, 8))
for i, name in enumerate(precursor_names):
    group = df_clean[df_clean[precursor_name_col] == name]
    ax.scatter(
        group.index,  # X轴：样本序号
        group['SC'],  # Y轴：比容量
        c=[colors[i % len(colors)]],  # 颜色循环使用
        marker=markers[i % len(markers)],  # 标记循环使用
        s=100,  # 点大小
        alpha=0.8,  # 透明度（避免重叠遮挡）
        edgecolors='black',
        linewidth=0.6,
        label=f'{name}'
    )
ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
ax.set_ylabel('Specific Capacity (SC)', fontsize=13, fontweight='bold')
# ax.set_title('Specific Capacity Distribution by Precursor Name (NAME Column)', fontsize=15, fontweight='bold', pad=20)
# ---------- 图例：放到底部，多列显示，避免超出画布 ----------
handles, labels = ax.get_legend_handles_labels()
ncol = 4  # 你可以改成 3/5/6，类别多就调大一点
ax.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),  # 往下放
    ncol=ncol,
    fontsize=9,
    framealpha=0.95,
    columnspacing=0.8,
    handletextpad=0.4,
    borderaxespad=0.2
)

# 给底部留出空间，不然图例会被裁剪
plt.tight_layout(rect=[0, 0.08, 1, 1])
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)

# 保存SC分布图
plt.tight_layout()
plt.savefig(output_path_sc, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"SC（比容量）分布图已保存至：{output_path_sc}")

# ========== 4.2 生成SE（比能量）独立分布图 ==========
fig, ax = plt.subplots(figsize=(12, 8))
for i, name in enumerate(precursor_names):
    group = df_clean[df_clean[precursor_name_col] == name]
    ax.scatter(
        group.index,  # X轴：样本序号
        group['SE'],  # Y轴：比能量
        c=[colors[i % len(colors)]],
        marker=markers[i % len(markers)],
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.6,
        label=f'{name}'
    )
ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
ax.set_ylabel('Specific Energy (SE)', fontsize=13, fontweight='bold')
ax.set_title('Specific Energy Distribution by Precursor Name (NAME Column)', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)

# 保存SE分布图
plt.tight_layout()
plt.savefig(output_path_se, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"SE（比能量）分布图已保存至：{output_path_se}")

# --------------------------
# 5. 可视化2：同名前驱体-综合性能关联图（核心）
# --------------------------
print("\n=== 步骤4：生成可视化图表2（综合性能关联） ===")
fig, ax = plt.subplots(figsize=(14, 10))

# 计算每个前驱体（NAME列）的平均性能
precursor_avg = df_clean.groupby(precursor_name_col)[target_features].mean().reset_index()
# 计算综合性能得分（用于后续排序）
precursor_avg['Overall_Score'] = precursor_avg['SC'] + precursor_avg['SE']

# 按综合得分排序，让性能好的前驱体在图例中靠前
precursor_avg_sorted = precursor_avg.sort_values('Overall_Score', ascending=False)
sorted_precursor_names = precursor_avg_sorted[precursor_name_col].tolist()

# 绘制散点（按综合性能排序的前驱体分组）
for i, name in enumerate(sorted_precursor_names):
    group = df_clean[df_clean[precursor_name_col] == name]
    # 获取该前驱体的平均性能
    avg_data = precursor_avg_sorted[precursor_avg_sorted[precursor_name_col] == name].iloc[0]
    avg_sc = avg_data['SC']
    avg_se = avg_data['SE']

    # 绘制样本点
    ax.scatter(
        group['SC'],  # X轴：比容量
        group['SE'],  # Y轴：比能量
        c=[colors[i % len(colors)]],
        marker=markers[i % len(markers)],
        s=120,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.8,
        label=f'{name} (n={len(group)}, Score={avg_data["Overall_Score"]:.2f})'  # 图例含综合得分
    )

    # 标注平均性能数值（带背景，清晰可读）
    ax.annotate(
        f'',
        xy=(avg_sc, avg_se),
        xytext=(12, 12),
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
        color='darkred',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='darkred')
    )

# 标注最优性能区域（红色箭头+背景）
max_sc = df_clean['SC'].max()
max_se = df_clean['SE'].max()
ax.annotate(
    'Optimal Performance Zone\n(High SC + High SE)',
    xy=(max_sc * 0.88, max_se * 0.88),
    xytext=(max_sc * 0.35, max_se * 0.55),
    arrowprops=dict(arrowstyle='->', color='darkred', lw=3, alpha=0.9, shrinkA=5, shrinkB=5),
    fontsize=18, color='darkred',fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='grey', alpha=0.2, edgecolor='darkred')
)

# 图表标签与标题（突出NAME列分类）
ax.set_xlabel('Specific Capacity (SC)', fontsize=15, fontweight='bold')
ax.set_ylabel('Specific Energy (SE)', fontsize=15, fontweight='bold')
# ax.set_title(
#     'Supercapacitor Precursor-Overall Performance Correlation\n(Grouped by NAME Column, Red Stars = Average Performance)',
#     fontsize=17, fontweight='bold', pad=30)
# 图例放在右侧，按综合得分排序
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)

# 保存图片（高分辨率，适合报告使用）
plt.tight_layout()
plt.savefig(output_path_2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"图表2（综合性能关联）已保存至：{output_path_2}")

# --------------------------
# 6. 输出最优前驱体分析结果（按NAME列）
# --------------------------
print("\n=== 步骤5：最优前驱体分析结果（按NAME列） ===")
print("按综合性能（比容量+比能量）排序的TOP5前驱体：")
for rank, (_, row) in enumerate(precursor_avg_sorted.head(5).iterrows(), 1):
    name = row[precursor_name_col]
    avg_sc = row['SC']
    avg_se = row['SE']
    score = row['Overall_Score']
    sample_count = len(df_clean[df_clean[precursor_name_col] == name])
    # 计算该前驱体的性能稳定性（SC的标准差，越小越稳定）
    sc_std = df_clean[df_clean[precursor_name_col] == name]['SC'].std()
    print(f"\n第{rank}名：{name}")
    print(f"  - 核心性能：平均比容量 {avg_sc:.3f} | 平均比能量 {avg_se:.3f} | 综合得分 {score:.3f}")
    print(f"  - 样本信息：{sample_count} 个样本 | 比容量标准差 {sc_std:.3f}（越小越稳定）")

best_precursor = precursor_avg_sorted.iloc[0][precursor_name_col]
best_score = precursor_avg_sorted.iloc[0]['Overall_Score']
print(f"\n🏆 综合性能最优的前驱体：{best_precursor}（综合得分：{best_score:.3f}）")
print("\n=== 所有分析完成！图表和结果文件已保存到指定路径 ===")

# --------------------------
# 7. 导出绘图专用 Excel 文件（供 Origin 使用）
# --------------------------
print("\n=== 步骤6：导出绘图专用Excel文件（Origin兼容） ===")

# 构建绘图数据表
plot_data = df_clean[[precursor_name_col, 'SC', 'SE']].copy()
plot_data.rename(columns={precursor_name_col: 'Precursor'}, inplace=True)
plot_data.reset_index(drop=True, inplace=True)

# 添加平均性能作为参考（可选：用于Origin中手动添加星标）
avg_data_for_excel = precursor_avg_sorted[[precursor_name_col, 'SC', 'SE', 'Overall_Score']].copy()
avg_data_for_excel.rename(columns={
    precursor_name_col: 'Precursor',
    'SC': 'Avg_SC',
    'SE': 'Avg_SE',
    'Overall_Score': 'Overall_Score'
}, inplace=True)

# 创建Excel写入器（多sheet）
excel_output_path = 'Precursor_PlotData_for_Origin.xlsx'
with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
    plot_data.to_excel(writer, sheet_name='Raw_Samples', index=False)
    avg_data_for_excel.to_excel(writer, sheet_name='Average_Performance', index=False)

print(f"✅ 绘图数据已导出至：{excel_output_path}")
print("  - Sheet 'Raw_Samples'：包含每个样本的 Precursor、SC、SE（用于散点图）")
print("  - Sheet 'Average_Performance'：包含各前驱体平均SC/SE及综合得分（用于标注/排序）")