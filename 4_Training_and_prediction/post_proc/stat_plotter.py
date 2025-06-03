import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV, handling potential ^M (carriage return) characters
df = pd.read_csv("validation_r2.csv", encoding="utf-8", engine="python")
df.columns = df.columns.str.strip()

# Clean up Station names and drop rows with missing R² scores
df["Station"] = df["Station"].str.strip()
df["Target"] = df["Target"].str.strip()
df = df[df["Validation R^2 Score"] != "N/A"]

# Convert R² score to float
df["Validation R^2 Score"] = df["Validation R^2 Score"].astype(float)

# Pivot the DataFrame to create a matrix
pivot_df = df.pivot(index="Target", columns="Station", values="Validation R^2 Score")

# Create the heatmap with larger fonts and higher resolution
plt.figure(figsize=(18, 12))  # Increase figure size for larger poster
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="BrBG", cbar_kws={'label': 'Validation R²'},
            linewidths=0.5, annot_kws={"size": 16})  # Increase font size for annotations

plt.title("Validation R² Scores Matrix", fontsize=20)  # Larger title font
plt.ylabel("Target Taxa", fontsize=16)  # Larger ylabel font
plt.xlabel("Station", fontsize=16)  # Larger xlabel font
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('val_R2_matrix.png')
# Increase resolution for high-quality printing
plt.tight_layout()
plt.savefig('val_R2_matrix.png', dpi=300)  # Set DPI for high resolution
plt.show()
