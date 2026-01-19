import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Taking Office' and 'Leaving' to integers, replacing 'Current' with 2024
df['Taking Office'] = df['Taking Office'].astype(int)
df['Leaving'] = df['Leaving'].replace('Current', 2024).astype(int)

# Calculate years served for each term
df['Years Served'] = df['Leaving'] - df['Taking Office']

# Sum all years served
total_years = df['Years Served'].sum()
print(f"Final Answer: {total_years}")