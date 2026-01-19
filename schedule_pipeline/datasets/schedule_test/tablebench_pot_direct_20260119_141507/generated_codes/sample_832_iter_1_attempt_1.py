import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built year' to integer
df['built year'] = pd.to_numeric(df['built year'], errors='coerce')

# Filter warships built before 1870
filtered_df = df[df['built year'] < 1870]

# Extract horse-power values for the filtered rows
horse_power_values = filtered_df['horse - power']

# Calculate standard deviation
std_horse_power = horse_power_values.std()

print(f"Final Answer: {std_horse_power:.1f}")