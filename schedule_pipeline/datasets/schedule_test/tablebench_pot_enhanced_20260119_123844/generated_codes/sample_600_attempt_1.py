import pandas as pd

df = pd.read_csv('table.csv')
# Extract and convert 'floors' to integer
df['floors'] = df['floors'].astype(int)
# Sort by height (in feet) in descending order
df['height_ft'] = df['height ft / m'].str.extract('(\d+)').astype(int)
top_5 = df.nlargest(5, 'height_ft')
average_floors = top_5['floors'].mean()
print(f"Final Answer: {average_floors:.1f}")