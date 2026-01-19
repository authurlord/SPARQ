import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'percentage lost' to float by removing the '%' sign
df['percentage lost'] = df['percentage lost'].str.replace('%', '').astype(float)

# Convert 'starting weight' to float (ensure correct type)
df['starting weight (kg)'] = df['starting weight (kg)'].astype(float)

# Calculate correlation between starting weight and percentage lost
correlation = df['starting weight (kg)'].corr(df['percentage lost'])

print(f"Final Answer: {correlation:.3f}")