import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'diameter (km)' column and convert to float
diameters = pd.to_numeric(df['diameter (km)'], errors='coerce')
# Remove any invalid entries (e.g., non-numeric)
diameters = diameters.dropna()
# Compute the median
median_diameter = diameters.median()
print(f"Final Answer: {median_diameter:.1f}")