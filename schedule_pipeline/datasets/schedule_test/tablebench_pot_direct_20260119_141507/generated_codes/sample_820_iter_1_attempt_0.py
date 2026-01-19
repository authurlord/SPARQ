import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'samples taken' to numeric, coercing errors to NaN
df['samples taken'] = pd.to_numeric(df['samples taken'], errors='coerce')

# Filter rows where samples taken >= 5
filtered_df = df[df['samples taken'] >= 5]

# Calculate average melamine content for filtered rows
avg_melamine = filtered_df['melamine content (mg / kg)'].mean()

print(f"Final Answer: {avg_melamine:.2f}")