import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract the feet value from 'height ft / m' column
df['height_ft'] = df['height ft / m'].str.extract(r'(\d+)/').astype(float)

# Sort by height in descending order and get top 5
top_5 = df.nlargest(5, 'height_ft')[['floors']]

# Calculate the average number of floors
avg_floors = top_5['floors'].mean()

print(f"Final Answer: {avg_floors:.1f}")