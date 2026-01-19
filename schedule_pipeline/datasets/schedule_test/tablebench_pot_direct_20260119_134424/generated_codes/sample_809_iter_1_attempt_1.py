import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'floors' column to integer for numerical comparison
df['floors'] = pd.to_numeric(df['floors'], errors='coerce')
# Filter buildings with more than 10 floors
filtered_df = df[df['floors'] > 10]
# Calculate mean height of these buildings
mean_height = filtered_df['height'].mean()
print(f"Final Answer: {mean_height:.1f}")