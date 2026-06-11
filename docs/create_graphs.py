import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(r'c:\Users\huzai\Downloads\benchmark_data_p\media_listing.csv')
df['Duration_Min'] = df['Duration_Min'].astype(float)
df['Size_MB'] = df['Size_MB'].astype(float)

# Define duration categories
bins = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ['0-1', '1-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
df['Duration_Category'] = pd.cut(df['Duration_Min'], bins=bins, labels=labels, right=False)

fig = plt.figure(figsize=(20, 24))
fig.suptitle('Media Files Dataset Analysis', fontsize=18, fontweight='bold', y=0.98)

# --- Graph 1: Bar chart - File count per duration category ---
ax1 = fig.add_subplot(3, 2, 1)
cat_counts = df['Duration_Category'].value_counts().sort_index()
bars = ax1.bar(cat_counts.index.astype(str), cat_counts.values, color='steelblue', edgecolor='black')
ax1.set_xlabel('Duration Category (minutes)')
ax1.set_ylabel('Number of Files')
ax1.set_title('Files per Duration Category')
for bar, val in zip(bars, cat_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontsize=9)
ax1.set_ylim(0, max(cat_counts.values) * 1.1)

# --- Graph 2: Pie chart - Duration category distribution ---
ax2 = fig.add_subplot(3, 2, 2)
colors = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
wedges, texts, autotexts = ax2.pie(cat_counts.values, labels=cat_counts.index.astype(str), 
                                    autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Duration Category Distribution (%)')

# --- Graph 3: Bar chart - File count per source category (folder) ---
ax3 = fig.add_subplot(3, 2, 3)
folder_counts = df['Folder'].value_counts().sort_values(ascending=True)
ax3.barh(folder_counts.index, folder_counts.values, color='coral', edgecolor='black')
ax3.set_xlabel('Number of Files')
ax3.set_ylabel('Source Category')
ax3.set_title('Files per Source Category')
for i, val in enumerate(folder_counts.values):
    ax3.text(val + 0.5, i, str(val), va='center', fontsize=8)

# --- Graph 4: Total duration per source category ---
ax4 = fig.add_subplot(3, 2, 4)
folder_duration = df.groupby('Folder')['Duration_Min'].sum().sort_values(ascending=True)
ax4.barh(folder_duration.index, folder_duration.values, color='mediumseagreen', edgecolor='black')
ax4.set_xlabel('Total Duration (minutes)')
ax4.set_ylabel('Source Category')
ax4.set_title('Total Duration per Source Category')
for i, val in enumerate(folder_duration.values):
    ax4.text(val + 1, i, f'{val:.0f} min', va='center', fontsize=8)

# --- Graph 5: Scatter plot - Size vs Duration ---
ax5 = fig.add_subplot(3, 2, 5)
folders = df['Folder'].unique()
cmap = plt.cm.tab20(np.linspace(0, 1, len(folders)))
for i, folder in enumerate(sorted(folders)):
    subset = df[df['Folder'] == folder]
    ax5.scatter(subset['Duration_Min'], subset['Size_MB'], label=folder, 
                alpha=0.7, s=30, color=cmap[i])
ax5.set_xlabel('Duration (minutes)')
ax5.set_ylabel('File Size (MB)')
ax5.set_title('File Size vs Duration (by Source)')
ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)

# --- Graph 6: Stacked bar - Source categories within each duration bucket ---
ax6 = fig.add_subplot(3, 2, 6)
pivot = df.groupby(['Duration_Category', 'Folder']).size().unstack(fill_value=0)
pivot.plot(kind='bar', stacked=True, ax=ax6, colormap='tab20', edgecolor='black', linewidth=0.3)
ax6.set_xlabel('Duration Category (minutes)')
ax6.set_ylabel('Number of Files')
ax6.set_title('Source Breakdown within Duration Categories')
ax6.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('media_distribution_graphs.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: media_distribution_graphs.png")

# --- Additional: Total size per source category ---
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Storage Analysis', fontsize=14, fontweight='bold')

# Total size per folder
folder_size = (df.groupby('Folder')['Size_MB'].sum() / 1024).sort_values(ascending=True)  # Convert to GB
axes[0].barh(folder_size.index, folder_size.values, color='mediumpurple', edgecolor='black')
axes[0].set_xlabel('Total Size (GB)')
axes[0].set_title('Total Storage per Source Category')
for i, val in enumerate(folder_size.values):
    axes[0].text(val + 0.1, i, f'{val:.1f} GB', va='center', fontsize=8)

# Average file size per folder
folder_avg = (df.groupby('Folder')['Size_MB'].mean()).sort_values(ascending=True)
axes[1].barh(folder_avg.index, folder_avg.values, color='goldenrod', edgecolor='black')
axes[1].set_xlabel('Average File Size (MB)')
axes[1].set_title('Average File Size per Source Category')
for i, val in enumerate(folder_avg.values):
    axes[1].text(val + 1, i, f'{val:.0f} MB', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('media_storage_graphs.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: media_storage_graphs.png")
print("\nDone! Two graph files created.")
