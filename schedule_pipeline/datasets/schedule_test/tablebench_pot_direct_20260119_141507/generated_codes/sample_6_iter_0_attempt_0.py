import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Taking Office' and 'Leaving' to integers
df['Taking Office'] = pd.to_numeric(df['Taking Office'], errors='coerce')
df['Leaving'] = pd.to_numeric(df['Leaving'], errors='coerce')

# For rows where 'Leaving' is 'Current', use 2024 as the leaving year
df['Leaving'] = df['Leaving'].fillna(2024)

# Calculate years served for each mayor
years_served = df['Leaving'] - df['Taking Office']

# Sum all years served
total_years = years_served.sum()

print(f"Final Answer: {total_years}")