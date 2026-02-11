import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file
csv_path = "results/metrics/epoch300_n5000_props.csv"
df = pd.read_csv(csv_path)

# Remove invalid entries
df = df[df["qed"].notna()]

print("Loaded:", len(df), "valid molecules")

# Create an output directory
os.makedirs("results/plots", exist_ok=True)

# ---------------------------
# 1. Histograms for each property
# ---------------------------

plt.figure(figsize=(6,4))
sns.histplot(df["qed"], bins=40, kde=True, color="skyblue")
plt.title("QED Distribution (Epoch 300)")
plt.xlabel("QED")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/plots/qed_hist.png")
plt.close()

plt.figure(figsize=(6,4))
sns.histplot(df["logp"], bins=40, kde=True, color="orange")
plt.title("logP Distribution (Epoch 300)")
plt.xlabel("logP")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/plots/logp_hist.png")
plt.close()

plt.figure(figsize=(6,4))
sns.histplot(df["sa"], bins=40, kde=True, color="green")
plt.title("SA Score Distribution (Epoch 300)")
plt.xlabel("SA Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/plots/sa_hist.png")
plt.close()

# ---------------------------
# 2. Scatter plots between properties
# ---------------------------

plt.figure(figsize=(6,6))
sns.scatterplot(x="logp", y="qed", data=df, alpha=0.3)
plt.title("QED vs logP")
plt.tight_layout()
plt.savefig("results/plots/qed_vs_logp.png")
plt.close()

plt.figure(figsize=(6,6))
sns.scatterplot(x="sa", y="qed", data=df, alpha=0.3)
plt.title("QED vs SA")
plt.tight_layout()
plt.savefig("results/plots/qed_vs_sa.png")
plt.close()

plt.figure(figsize=(6,6))
sns.scatterplot(x="logp", y="sa", data=df, alpha=0.3)
plt.title("logP vs SA")
plt.tight_layout()
plt.savefig("results/plots/logp_vs_sa.png")
plt.close()

# ---------------------------
# 3. Pairplot (joint distribution)
# ---------------------------

sns.pairplot(df[["qed", "logp", "sa"]], diag_kind="kde")
plt.savefig("results/plots/pairplot.png")
plt.close()

print("\nAll plots saved to results/plots/")
