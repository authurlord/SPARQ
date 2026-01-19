import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])
# Calculate the difference (increase) from the previous year
df['increase'] = df['total s ton'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")