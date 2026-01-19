import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'samples taken' to integer to allow comparison
df['samples taken'] = pd.to_numeric(df['samples taken'], errors='coerce')
# Filter rows where samples taken is at least 5
filtered_df = df[df['samples taken'] >= 5]
# Calculate average melamine content for these rows
average_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {average_melamine:.2f}")