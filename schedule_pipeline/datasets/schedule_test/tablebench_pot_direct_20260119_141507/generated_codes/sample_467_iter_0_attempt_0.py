import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert population and other numeric columns to numeric if needed
df['average annual rainfall (mm)'] = pd.to_numeric(df['average annual rainfall (mm)'], errors='coerce')
df['average annual runoff (mm)'] = pd.to_numeric(df['average annual runoff (mm)'], errors='coerce')

# Remove rows with missing values
df_clean = df.dropna(subset=['average annual rainfall (mm)', 'average annual runoff (mm)'])

# Calculate mean and std for rainfall and runoff
mean_rainfall = df_clean['average annual rainfall (mm)'].mean()
std_rainfall = df_clean['average annual rainfall (mm)'].std()

mean_runoff = df_clean['average annual runoff (mm)'].mean()
std_runoff = df_clean['average annual runoff (mm)'].std()

# Define outlier thresholds (2 standard deviations)
rainfall_upper = mean_rainfall + 2 * std_rainfall
rainfall_lower = mean_rainfall - 2 * std_rainfall
runoff_upper = mean_runoff + 2 * std_runoff
runoff_lower = mean_runoff - 2 * std_runoff

# Identify regions that are outliers in either rainfall or runoff
outliers = []
for idx, row in df_clean.iterrows():
    rainfall = row['average annual rainfall (mm)']
    runoff = row['average annual runoff (mm)']
    
    if (rainfall > rainfall_upper or rainfall < rainfall_lower or
        runoff > runoff_upper or runoff < runoff_lower):
        outliers.append(row['administrative region'])

# Remove duplicates and print result
outlier_regions = list(set(outliers))
print(f"Final Answer: {', '.join(outlier_regions)}")