import pandas as pd

df = pd.read_csv('table.csv')
# Filter buildings with more than 10 floors
filtered_df = df[df['floors'] > 10]
# Convert 'height' to numeric for calculation
filtered_df['height'] = pd.to_numeric(filtered_df['height'])
# Calculate mean height
mean_height = filtered_df['height'].mean()
print(f"Final Answer: {mean_height:.1f}")