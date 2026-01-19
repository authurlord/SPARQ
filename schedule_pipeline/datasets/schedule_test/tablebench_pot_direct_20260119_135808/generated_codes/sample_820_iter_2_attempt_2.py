import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'samples taken' to integer and filter rows with at least 5 samples
filtered_df = df[df['samples taken'].astype(int) >= 5]
# Calculate the average melamine content for the filtered data
average_melamine = filtered_df['melamine content (mg / kg)'].mean()
print(f"Final Answer: {average_melamine:.2f}")