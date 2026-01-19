import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'average relative annual growth (%)' to numeric, coercing errors to NaN
df['average relative annual growth (%)'] = pd.to_numeric(df['average relative annual growth (%)'], errors='coerce')

# Drop rows with missing growth data
df = df.dropna(subset=['average relative annual growth (%)'])

# Extract growth rates
growth_rates = df['average relative annual growth (%)'].astype(float)

# Compute mean and standard deviation
mean_growth = growth_rates.mean()
std_growth = growth_rates.std()

# Define threshold for outliers (2 standard deviations)
threshold = 2 * std_growth

# Identify outliers: values more than 2 std away from mean
outliers = np.abs(growth_rates - mean_growth) > threshold

# Get country names of outliers
outlier_countries = df.loc[outliers, 'country (or dependent territory)'].tolist()

print(f"Final Answer: {', '.join(outlier_countries)}")