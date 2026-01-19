import pandas as pd

df = pd.read_csv('table.csv')
# Replace 'Current' with 2024 for calculation
df['Leaving'] = df['Leaving'].replace('Current', 2024)
# Convert columns to integers
df['Taking Office'] = pd.to_numeric(df['Taking Office'])
df['Leaving'] = pd.to_numeric(df['Leaving'])
# Calculate years served
df['Years Served'] = df['Leaving'] - df['Taking Office']
# Sum total years served
total_years = df['Years Served'].sum()
print(f"Final Answer: {total_years}")