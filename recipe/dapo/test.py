# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# def describe(arr):
#     if len(arr) == 0:
#         return {"count": 0}
#     return {
#         "count": len(arr),
#         "mean": np.mean(arr),
#         "median": np.median(arr),
#         "min": np.min(arr),
#         "max": np.max(arr),
#     }

# # ========= 1. 读取数据 =========
# df = pd.read_parquet(
#     "/data2/xucaijun/Data-Select-verl/ckpts/Epoch-filter/entropy-d-True-s-False-b-192-64-64-1708-dataset-think-DeepMath-1000-model-Qwen2.5-7B/epoch_7_data.parquet"
# )
# print(df.head())
# # ========= 2. 预分配容器 =========
# # 若每行 metric_list ⻓度不同，就取全局最大长度
# max_len = max(len(m) for m in df["metric_list"])
# # i 只需要到 len-2，因为要看 i 与 i+1
# entropy_eq8_all  = [[] for _ in range(max_len - 1)]
# entropy_not8_all = [[] for _ in range(max_len - 1)]

# # ========= 3. 遍历收集熵 =========
# for metrics, entropies,prompt in zip(df["metric_list"], df["entropy_list"],df['prompt']):
#     seq_len = len(metrics)
#     for i in range(seq_len - 1):
#         if metrics[i] == 8:
#             prompt_len= len(prompt[1]['content'])
#             entropy= entropies[i] 
#             if metrics[i + 1] == 8:
#                 entropy_eq8_all[i].append(entropy)
#             else:
#                 entropy_not8_all[i].append(entropy)

# # ========= 4. 逐步画图并保存 =========
# save_dir = Path("./entropy_transition_figs")
# save_dir.mkdir(parents=True, exist_ok=True)

# for i in range(max_len - 1):
#     # 如果该步没有任何数据，跳过
#     if not entropy_eq8_all[i] and not entropy_not8_all[i]:
#         continue

#     plt.figure()
#     plt.hist(
#         [entropy_not8_all[i], entropy_eq8_all[i]],
#         bins=100,              # 想要不同箱数可改
#         stacked=True,
#         label=["metric 8 → non-8", "metric 8 → 8"],
#     )
#     plt.xlabel("Entropy value")
#     plt.ylabel("Count (stacked)")
#     plt.title(f"Entropy Distribution at Step {i} → {i+1}")
#     plt.legend()
#     plt.tight_layout()

#     out_path = save_dir / f"entropy_step_{i}_to_{i+1}.png"
#     plt.savefig(out_path)
#     plt.close()   # 释放内存
#     print(f"Saved: {out_path}")

#     stats_eq8 = describe(entropy_eq8_all[i])
#     stats_not8 = describe(entropy_not8_all[i])
#     print(f"Step {i} to {i+1} \nMetric 8 → 8: {stats_eq8}, \nMetric 8 → non-8: {stats_not8}")


# # ========= 5. 计算每步均值曲线并绘制 =========
# mean_eq8   = []        # metric 8 → 8 的均值
# mean_not8 = []         # metric 8 → non-8 的均值

# percentage_not8 = []  # metric 8 → non-8 的百分比
# steps      = []        # 有数据的 i

# for i in range(max_len - 1):
#     # 只统计至少有一类数据的步长
#     if entropy_eq8_all[i] or entropy_not8_all[i]:
#         steps.append(i)
#         # 若该类为空则填 NaN，绘图时会自动断开
#         mean_eq8.append(
#             np.mean(entropy_eq8_all[i]) if entropy_eq8_all[i] else np.nan
#         )
#         mean_not8.append(
#             np.mean(entropy_not8_all[i]) if entropy_not8_all[i] else np.nan
#         )

#         percentage_not8.append(
#             len(entropy_not8_all[i]) / (len(entropy_eq8_all[i]) + len(entropy_not8_all[i]))
#             if (len(entropy_eq8_all[i]) + len(entropy_not8_all[i])) > 0 else np.nan
#         )

# plt.figure()
# plt.plot(steps, mean_not8, marker="o", label="metric 8 → non-8 (mean)")
# plt.plot(steps, mean_eq8,  marker="s", label="metric 8 → 8 (mean)")
# plt.xlabel("Step i")
# plt.ylabel("Mean entropy at i+1")
# plt.title("Mean Entropy Transition Curve")
# plt.legend()
# plt.tight_layout()

# out_path = save_dir / "entropy_mean_vs_step.png"
# plt.savefig(out_path)
# plt.close()
# print(f"Saved: {out_path}")



# plt.figure()
# plt.plot(steps, percentage_not8, marker="o")
# plt.xlabel("Step i")
# plt.ylabel("Non 8 percentage at i")
# plt.legend()
# plt.tight_layout()

# out_path = save_dir / "Non_8_percentage.png"
# plt.savefig(out_path)
# plt.close()
# print(f"Saved: {out_path}")

print([1]*10)