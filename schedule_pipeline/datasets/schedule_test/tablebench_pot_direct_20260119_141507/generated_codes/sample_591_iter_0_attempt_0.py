import pandas as pd

df = pd.read_csv('table.csv')
# Select live births for the specified periods
live_births = df.loc[(df['Period'].str.startswith('1950')) | 
                     (df['Period'].str.startswith('1955')) | 
                     (df['Period'].str.startswith('1960')) | 
                     (df['Period'].str.startswith('1965')) | 
                     (df['Period'].str.startswith('1970')) | 
                     (df['Period'].str.startswith('1975')) | 
                     (df['Period'].str.startswith('1980')), 'Live births per year']

# Convert to numeric and sum
total_live_births = live_births.str.replace(' ', '').astype(float).sum()
print(f"Final Answer: {total_live_births}")