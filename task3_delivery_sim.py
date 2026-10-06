import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
# Creating 100 synthetic delivery times
delivery_times = np.random.normal(
    loc=31.5,
    scale=5.5,
    size=97
)
delivery_times = np.round(delivery_times).astype(int)
delivery_times = np.maximum(delivery_times, 10)
# Adding 3 unusually long deliveries
delivery_times = np.concatenate([
    delivery_times,
    [68, 71, 75]
])
df_source = pd.DataFrame({
    "OrderID": np.arange(1, 101),
    "DeliveryTime(min)": delivery_times
})
# Save CSV
df_source.to_csv("delivery_times.csv", index=False)
df = pd.read_csv("delivery_times.csv")
real_times = df["DeliveryTime(min)"]
# (a) Mean and standard deviation
mean_time = real_times.mean()
std_time = real_times.std()
print(
    f"Real delivery times -- Mean: "
    f"{mean_time:.2f} min "
    f"Std: {std_time:.2f} min"
)
# (b) Simulating 100 delivery times using same mean and std
simulated_times = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)
print(
    f"Simulated 100 values from "
    f"Normal(mean={mean_time:.2f}, "
    f"std={std_time:.2f})"
)
# (d) Missing values
print("\nMissing values:", df.isnull().sum().sum())
# Percentiles
q1 = real_times.quantile(0.25)
median = real_times.quantile(0.50)
q3 = real_times.quantile(0.75)
print(
    f"Percentiles -- "
    f"25th: {q1:.2f} "
    f"50th: {median:.2f} "
    f"75th: {q3:.2f}"
)
# (e) IQR method
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
# Find outliers
outlier_mask = (
    (real_times < lower_bound) |
    (real_times > upper_bound)
)
outliers = df[outlier_mask]
print(
    f"IQR outlier bounds: "
    f"{lower_bound:.2f} - {upper_bound:.2f}"
)
print(
    f"Outliers detected: "
    f"{len(outliers)} deliveries "
    f"({outliers['DeliveryTime(min)'].tolist()} minutes)"
)
outlier_percentage = (
    len(outliers) / len(df)
) * 100
print(
    f"Percentage flagged as outliers: "
    f"{outlier_percentage:.2f}%"
)
# (c) Overlapping histograms
plt.figure(figsize=(8, 5))
plt.hist(
    real_times,
    bins=12,
    alpha=0.6,
    label="Real Delivery Times",
    edgecolor="black"
)
plt.hist(
    simulated_times,
    bins=12,
    alpha=0.6,
    label="Simulated Normal Times",
    edgecolor="black"
)
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig(
    "delivery_comparison_histogram.png",
    dpi=150
)
plt.close()
# Boxplot
plt.figure(figsize=(6, 5))
plt.boxplot(real_times)
plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")
plt.tight_layout()
plt.savefig(
    "delivery_boxplot.png",
    dpi=150
)
plt.close()
print(
    "\nSaved delivery_comparison_histogram.png, "
    "delivery_boxplot.png"
)