import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])
# Calculate year-on-year increase
df['increase'] = df['total s ton'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['increase'].idxmax()]
# Output the total s ton for that year
print(f"Final Answer: {max_increase_year['total s ton']}")