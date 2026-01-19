import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, handling any non-numeric values (e.g., spaces in numbers)
df['Average population (x 1000)'] = pd.to_numeric(df['Average population (x 1000)'], errors='coerce')
df['Natural change (per 1000)'] = pd.to_numeric(df['Natural change (per 1000)'], errors='coerce')

# Calculate correlation coefficient
correlation = df['Average population (x 1000)'].corr(df['Natural change (per 1000)'])
print(f"Final Answer: {correlation:.3f}")