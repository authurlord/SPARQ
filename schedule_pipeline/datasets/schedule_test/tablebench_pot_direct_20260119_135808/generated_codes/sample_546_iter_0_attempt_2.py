import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to integer
df['total s ton'] = df['total s ton'].astype(int)
# Calculate the year-on-year increase
df['increase'] = df['total s ton'].diff()
# Find the year with the maximum increase (excluding the first year)
max_increase_year = df.loc[df['increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")