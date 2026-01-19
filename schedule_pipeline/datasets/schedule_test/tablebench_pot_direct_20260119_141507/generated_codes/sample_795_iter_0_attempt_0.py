import pandas as pd

df = pd.read_csv('table.csv')

# Convert the passenger traffic columns to numeric
df[['2008', '2009', '2010']] = df[['2008', '2009', '2010']].apply(pd.to_numeric, errors='coerce')

# Compute average passenger traffic per year
year_averages = df[['2008', '2009', '2010']].mean(axis=0)

# Find the year with the highest average
max_year = year_averages.idxmax()
max_avg = year_averages[max_year]

# Compute standard deviation for that year
std_dev = df[max_year].std()

print(f"Final Answer: {max_year}, {std_dev:.1f}")