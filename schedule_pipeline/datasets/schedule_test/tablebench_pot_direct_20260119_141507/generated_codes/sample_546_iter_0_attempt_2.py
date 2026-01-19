import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' to integer and sort by year
df['year'] = df['year'].astype(int)
df = df.sort_values('year')

# Extract total s ton and calculate year-over-year differences
total_s_ton = df['total s ton'].astype(int)
yearly_increase = total_s_ton.iloc[1:] - total_s_ton.iloc[:-1]

# Find the year with the highest increase
max_increase_index = yearly_increase.idxmax()
max_increase_year = df.loc[yearly_increase.idxmax(), 'year']

print(f"Final Answer: {max_increase_year}")