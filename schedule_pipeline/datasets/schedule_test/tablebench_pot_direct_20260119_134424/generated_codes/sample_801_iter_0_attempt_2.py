import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'diameter (km)' to numeric, handling any potential non-numeric values
df['diameter (km)'] = pd.to_numeric(df['diameter (km)'], errors='coerce')
# Drop any rows with NaN values in the diameter column
df.dropna(subset=['diameter (km)'], inplace=True)
# Calculate the median
median_diameter = df['diameter (km)'].median()
print(f"Final Answer: {median_diameter:.1f}")