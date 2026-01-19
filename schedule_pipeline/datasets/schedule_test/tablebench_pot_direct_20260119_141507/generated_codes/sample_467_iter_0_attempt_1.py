import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract rainfall and runoff columns
rainfall = df['average annual rainfall (mm)'].astype(float)
runoff = df['average annual runoff (mm)'].astype(float)

# Compute mean and standard deviation
mean_rainfall = rainfall.mean()
std_rainfall = rainfall.std()
mean_runoff = runoff.mean()
std_runoff = runoff.std()

# Define threshold for outliers (2 standard deviations)
threshold_rainfall = 2 * std_rainfall
threshold_runoff = 2 * std_runoff

# Identify regions with abnormal rainfall or runoff
outlier_regions = []
for idx, row in df.iterrows():
    region = row['administrative region']
    rain = float(row['average annual rainfall (mm)'])
    run = float(row['average annual runoff (mm)'])
    
    if abs(rain - mean_rainfall) > threshold_rainfall or abs(run - mean_runoff) > threshold_runoff:
        outlier_regions.append(region)

# Remove duplicates and sort
outlier_regions = list(set(outlier_regions))
outlier_regions.sort()

print(f"Final Answer: {', '.join(outlier_regions)}")