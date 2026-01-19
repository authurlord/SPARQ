import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Taking Office' and 'Leaving' to integers; use 2024 for 'Current'
df['Taking Office'] = pd.to_numeric(df['Taking Office'])
df['Leaving'] = df['Leaving'].replace('Current', 2024)
df['Leaving'] = pd.to_numeric(df['Leaving'])

# Calculate years served
df['Years Served'] = df['Leaving'] - df['Taking Office']

# Sum total years served
total_years = df['Years Served'].sum()
print(f"Final Answer: {total_years}")