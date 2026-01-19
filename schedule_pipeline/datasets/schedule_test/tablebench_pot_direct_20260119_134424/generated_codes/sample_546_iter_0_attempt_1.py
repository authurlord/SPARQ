import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to integer
df['total s ton'] = df['total s ton'].astype(int)
# Calculate year-over-year increase
df['increase'] = df['total s ton'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['increase'].idxmax()]
# Output the total s ton for that year
print(f"Final Answer: {max_increase_year['total s ton']}")