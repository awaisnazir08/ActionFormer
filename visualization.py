import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Your fine-tuned ActionFormer results
our_model = {
    'Model': 'ActionFormer',
    'F1@50%': 84.15,
    'F1@25%': 86.34,
    'F1@10%': 86.34,
    'Accuracy': 81.05
}

# Other benchmarks from literature
benchmarks = [
    {'Model': 'BaFormer',   'F1@50%': 83.5,  'F1@25%': 91.3, 'F1@10%': 92.0, 'Accuracy': 83.0},
    {'Model': 'ASFormer',   'F1@50%': 79.2,  'F1@25%': 88.8, 'F1@10%': 90.1, 'Accuracy': 79.7},
    {'Model': 'MS-TCN++',   'F1@50%': 76.0,  'F1@25%': 85.7, 'F1@10%': 88.8, 'Accuracy': 80.1},
    {'Model': 'MS-TCN',     'F1@50%': 74.6,  'F1@25%': 85.4, 'F1@10%': 87.5, 'Accuracy': 79.2},
    our_model
]

# Convert to DataFrame
df = pd.DataFrame(benchmarks)

# Melt to long-form for seaborn
df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))
palette = sns.color_palette("Paired", len(df["Model"].unique()))

ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette=palette)
plt.title("Comparison of Action Segmentation Models on GTEA with Localization Model", fontsize=16)
plt.ylabel("Score (%)")
plt.ylim(70, 100)
plt.legend(title='Model', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()
