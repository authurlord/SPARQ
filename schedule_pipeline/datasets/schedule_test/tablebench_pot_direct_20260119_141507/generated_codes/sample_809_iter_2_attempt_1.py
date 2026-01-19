import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'height' to numeric, coercing errors to NaN (though all should be valid)
df['height'] = pd.to_numeric(df['height'], errors='coerce')

# Filter buildings with more than 10 floors
filtered_buildings = df[df['floors'] > 10]

# Calculate mean height of filtered buildings
mean_height = filtered_buildings['height'].mean()
print(f"Final Answer: {mean_height:.1f}")