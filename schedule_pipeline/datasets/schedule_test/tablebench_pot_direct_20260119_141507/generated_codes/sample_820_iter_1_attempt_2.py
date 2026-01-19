import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'samples taken' to numeric, coercing errors to NaN
df['samples taken'] = pd.to_numeric(df['samples taken'], errors='coerce')

# Filter rows where samples taken >= 5
filtered_df = df[df['samples taken'] >= 5]

# Calculate the average melamine content for the filtered rows
average_melamine = filtered_df['melamine content (mg / kg)'].mean()

print(f"Final Answer: {average_melamine:.2f}")