import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of tropical storms' to integer
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'])
# Calculate the difference in tropical storms from the previous year
df['increase'] = df['number of tropical storms'].diff()
# Find the year with the maximum increase
max_increase_year = df.loc[df['increase'].idxmax(), 'year']
print(f"Final Answer: {max_increase_year}")