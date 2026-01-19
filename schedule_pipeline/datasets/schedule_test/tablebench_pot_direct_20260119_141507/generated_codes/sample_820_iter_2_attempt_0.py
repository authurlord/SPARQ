import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'samples taken' is at least 5
filtered_df = df[df['samples taken'] >= 5]
# Calculate the mean of melamine content for these rows
mean_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {mean_melamine:.2f}")