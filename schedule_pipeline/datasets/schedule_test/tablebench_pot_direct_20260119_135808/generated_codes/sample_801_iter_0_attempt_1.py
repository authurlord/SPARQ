import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'diameter (km)' to numeric
df['diameter (km)'] = pd.to_numeric(df['diameter (km)'])
# Calculate median
median_diameter = df['diameter (km)'].median()
print(f"Final Answer: {median_diameter}")