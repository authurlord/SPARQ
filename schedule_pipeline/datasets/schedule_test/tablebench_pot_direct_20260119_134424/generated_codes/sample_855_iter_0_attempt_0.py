import pandas as pd

df = pd.read_csv('table.csv')
# Convert the required columns to integers
df['2005 - 06'] = pd.to_numeric(df['2005 - 06'])
df['2008 - 09'] = pd.to_numeric(df['2008 - 09'])

# Calculate the absolute difference between '2005 - 06' and '2008 - 09'
df['difference'] = (df['2005 - 06'] - df['2008 - 09']).abs()

# Sort by difference and get the two stations with the smallest difference
top_two = df.nsmallest(2, 'difference')['station name']

print(f"Final Answer: {top_two.iloc[0]}, {top_two.iloc[1]}")