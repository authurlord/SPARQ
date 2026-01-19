import pandas as pd

df = pd.read_csv('table.csv')
# Convert the years columns to numeric
years = ['2008', '2009', '2010']
df[years] = df[years].apply(pd.to_numeric)

# Calculate average passenger traffic per year
avg_traffic = df[years].mean()

# Find the year with the highest average
max_avg_year = avg_traffic.idxmax()
max_avg_value = avg_traffic.max()

# Calculate standard deviation for that year
std_dev = df[max_avg_year].std()

print(f"Final Answer: {max_avg_year}, {std_dev:.1f}")