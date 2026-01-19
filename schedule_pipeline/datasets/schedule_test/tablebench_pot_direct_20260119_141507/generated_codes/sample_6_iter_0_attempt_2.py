import pandas as pd

df = pd.read_csv('table.csv')

# Convert Taking Office and Leaving to integers
def parse_year(year_str):
    if year_str == 'Current':
        return 2024
    return int(year_str)

df['Taking Office'] = df['Taking Office'].apply(parse_year)
df['Leaving'] = df['Leaving'].apply(parse_year)

# Calculate years served
df['years_served'] = df['Leaving'] - df['Taking Office']

# Sum total years served
total_years = df['years_served'].sum()

print(f"Final Answer: {total_years}")