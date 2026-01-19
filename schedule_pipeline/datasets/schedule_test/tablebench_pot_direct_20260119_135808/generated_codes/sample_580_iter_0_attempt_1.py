import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bötzow' column to float
df['bötzow'] = pd.to_numeric(df['bötzow'])
# Calculate the year-on-year increase
df['increase'] = df['bötzow'].diff()
# Find the year with the maximum increase (excluding the first year)
max_increase_year = df.loc[df['increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")