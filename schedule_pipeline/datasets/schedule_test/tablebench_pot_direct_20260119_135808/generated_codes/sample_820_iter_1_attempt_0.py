import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'samples taken' to integer to avoid type mismatch
df['samples taken'] = pd.to_numeric(df['samples taken'], errors='coerce')
# Filter rows where samples taken >= 5
filtered_df = df[df['samples taken'] >= 5]
# Calculate average melamine content
average_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {average_melamine:.2f}")