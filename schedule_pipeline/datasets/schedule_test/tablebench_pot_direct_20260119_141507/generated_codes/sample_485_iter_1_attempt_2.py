import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop rows with missing values in 'Top speed' or 'Number'
df_clean = df.dropna(subset=['Top speed_1', 'Number'])

# Convert 'Top speed' to numeric; use 'mph' column since it's in consistent units
df_clean['Top speed'] = pd.to_numeric(df_clean['Top speed'], errors='coerce')
df_clean = df_clean.dropna(subset=['Top speed'])

# Extract relevant columns
top_speed = df_clean['Top speed'].astype(float)
number_units = df_clean['Number'].astype(float)

# Calculate mean and std
mean_top_speed = top_speed.mean()
std_top_speed = top_speed.std()

mean_number = number_units.mean()
std_number = number_units.std()

# Identify outliers using 2 standard deviations
outlier_top_speed = np.abs(top_speed - mean_top_speed) > 2 * std_top_speed
outlier_number = np.abs(number_units - mean_number) > 2 * std_number

# Get classes that are outliers in either top speed or number
outlier_classes = []
for idx, row in df_clean.iterrows():
    class_name = row['Class']
    if outlier_top_speed[idx] or outlier_number[idx]:
        outlier_classes.append(class_name)

# Remove duplicates and sort
outlier_classes = list(set(outlier_classes))
outlier_classes.sort()

print(f"Final Answer: {', '.join(outlier_classes)}")