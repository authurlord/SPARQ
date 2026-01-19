import pandas as pd

df = pd.read_csv('table.csv')

# Extract the height in feet (first number before the slash)
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(int)

# Sort by height in descending order and get the top 5
top_5 = df.nlargest(5, 'height_ft')

# Calculate the average number of floors for the top 5
avg_floors = top_5['floors'].mean()

print(f"Final Answer: {avg_floors:.1f}")