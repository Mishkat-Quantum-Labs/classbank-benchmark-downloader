import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'c:\Users\huzai\Downloads\benchmark_data_p\media_listing.csv')
df['Duration_Min'] = df['Duration_Min'].astype(float)

bins = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ['0-1', '1-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
df['Category'] = pd.cut(df['Duration_Min'], bins=bins, labels=labels, right=False)

counts = df['Category'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(counts.index.astype(str), counts.values, color='#4C9BE8', edgecolor='black', width=0.6)

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val), ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Duration (minutes)', fontsize=12)
ax.set_ylabel('Number of Files', fontsize=12)
ax.set_title('Media Files Distribution by Duration', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\huzai\Downloads\benchmark_data_p\media_distribution_chart.png', dpi=150)
plt.close()
print("Saved: media_distribution_chart.png")
