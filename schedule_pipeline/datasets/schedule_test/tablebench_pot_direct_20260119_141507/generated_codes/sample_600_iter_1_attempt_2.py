import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Parse the 'height ft / m' column to extract feet (first number before '/')
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(int)

# Sort by height in descending order and get top 5
top_5 = df.sort_values(by='height_ft', ascending=False).head(5)

# Extract the 'floors' values from the top 5
floors_top_5 = top_5['floors'].astype(int)

# Compute the average
avg_floors = floors_top_5.mean()

print(f"Final Answer: {avg_floors:.1f}")