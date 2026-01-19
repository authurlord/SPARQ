import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to integer
df['total s ton'] = df['total s ton'].astype(int)
# Calculate year-on-year increase
df['increase'] = df['total s ton'].diff()
# Find the year with the maximum increase (skip first row as no previous year)
max_increase_year = df.loc[df['increase'].idxmax()]
# Return the total s ton of that year
print(f"Final Answer: {max_increase_year['total s ton']}")