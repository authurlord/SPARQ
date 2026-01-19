import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where samples taken is at least 5
filtered_df = df[df['samples taken'] >= 5]
# Calculate the mean of melamine content for the filtered rows
average_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {average_melamine:.2f}")