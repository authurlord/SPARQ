import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert area and population to numeric, handling any parsing errors
df['area (km 2)'] = pd.to_numeric(df['area (km 2)'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Drop rows with missing values
df_clean = df.dropna(subset=['area (km 2)', 'population'])

# Calculate mean and std for area and population
mean_area = df_clean['area (km 2)'].mean()
std_area = df_clean['area (km 2)'].std()
mean_pop = df_clean['population'].mean()
std_pop = df_clean['population'].std()

# Define Z-score threshold (2 standard deviations)
z_threshold = 2

# Find outliers in area
outliers_area = df_clean[(np.abs((df_clean['area (km 2)'] - mean_area) / std_area) > z_threshold)]
outliers_pop = df_clean[(np.abs((df_clean['population'] - mean_pop) / std_pop) > z_threshold)]

# Combine outlier locations
outlier_locations = set()
for idx, row in outliers_area.iterrows():
    outlier_locations.add(row['place'])
for idx, row in outliers_pop.iterrows():
    outlier_locations.add(row['place'])

# Convert to list and print
outlier_list = list(outlier_locations)
print(f"Final Answer: {', '.join(outlier_list)}")