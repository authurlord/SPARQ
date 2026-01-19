import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'samples taken' to integer to allow numerical comparison
df['samples taken'] = pd.to_numeric(df['samples taken'])
# Filter rows where samples taken is at least 5
filtered_df = df[df['samples taken'] >= 5]
# Calculate the average melamine content for the filtered data
average_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {average_melamine:.2f}")