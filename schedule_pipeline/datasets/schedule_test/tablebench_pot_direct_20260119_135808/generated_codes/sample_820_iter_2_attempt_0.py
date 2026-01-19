import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'samples taken' to integer and filter rows with at least 5 samples
df_filtered = df[df['samples taken'] >= 5]

# Calculate the average melamine content for the filtered rows
average_melamine = df_filtered['melamine content (mg / kg)'].mean()

print(f"Final Answer: {average_melamine:.2f}")