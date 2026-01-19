import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'floors' to integer type
df['floors'] = df['floors'].astype(int)
# Filter buildings with more than 10 floors
filtered_buildings = df[df['floors'] > 10]
# Calculate the mean height of the filtered buildings
mean_height = filtered_buildings['height'].mean()
print(f"Final Answer: {mean_height:.1f}")