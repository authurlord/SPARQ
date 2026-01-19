import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Taking Office' and 'Leaving' to integers
df['Taking Office'] = pd.to_numeric(df['Taking Office'])
df['Leaving'] = pd.to_numeric(df['Leaving'])

# Calculate years served for each mayor
df['Years Served'] = df['Leaving'] - df['Taking Office']

# Sum the total years served
total_years = df['Years Served'].sum()
print(f"Final Answer: {total_years}")