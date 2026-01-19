import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bötzow' column to float
df['bötzow'] = pd.to_numeric(df['bötzow'])
# Calculate the year-on-year difference
df['diff'] = df['bötzow'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['diff'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")