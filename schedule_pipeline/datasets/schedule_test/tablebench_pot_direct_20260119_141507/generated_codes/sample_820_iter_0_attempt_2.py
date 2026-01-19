import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where samples taken is at least 5
filtered_df = df[df['samples taken'] >= 5]
# Calculate the average melamine content for these filtered rows
avg_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {avg_melamine:.2f}")