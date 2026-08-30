import pandas as pd
import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# 1. LOAD FINAL DATASET
# --------------------------------------------------

file_path = "data/processed/final_aqi_dataset.csv"

df = pd.read_csv(file_path)

df["time"] = pd.to_datetime(df["time"])


# --------------------------------------------------
# 2. BASIC DATASET INFORMATION
# --------------------------------------------------

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nTotal rows:", df.shape[0])
print("Total columns:", df.shape[1])


# --------------------------------------------------
# 3. SELECT IMPORTANT COLUMNS
# --------------------------------------------------

features = [
    "AQI",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]


# --------------------------------------------------
# 4. STATISTICAL SUMMARY
# --------------------------------------------------

print("\nStatistical summary:")

print(
    df[features]
    .describe()
    .transpose()
)


# --------------------------------------------------
# 5. CORRELATION WITH AQI
# --------------------------------------------------

correlation = df[features].corr()

aqi_correlation = correlation["AQI"].sort_values(
    ascending=False
)

print("\nCorrelation with AQI:")

print(aqi_correlation)


# --------------------------------------------------
# 6. CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs(
    "outputs/plots",
    exist_ok=True
)


# --------------------------------------------------
# 7. AQI OVER TIME GRAPH
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["time"],
    df["AQI"]
)

plt.title("AQI Variation Over Time")

plt.xlabel("Time")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    "outputs/plots/aqi_over_time.png"
)

plt.close()


# --------------------------------------------------
# 8. AQI DISTRIBUTION GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["AQI"],
    bins=30
)

plt.title("AQI Distribution")

plt.xlabel("AQI")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/plots/aqi_distribution.png"
)

plt.close()


# --------------------------------------------------
# 9. AQI CATEGORY GRAPH
# --------------------------------------------------

category_counts = df["AQI_Category"].value_counts()

plt.figure(figsize=(9, 5))

category_counts.plot(
    kind="bar"
)

plt.title("AQI Category Distribution")

plt.xlabel("AQI Category")
plt.ylabel("Number of Records")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/plots/aqi_categories.png"
)

plt.close()


print("\nEDA completed successfully!")

print("\nGraphs saved inside:")
print("outputs/plots/")