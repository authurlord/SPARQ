import pandas as pd

df = pd.read_csv('table.csv')
# Convert the year columns to numeric
year_columns = ['2008', '2009', '2010']
df[year_columns] = df[year_columns].apply(pd.to_numeric)

# Calculate average passenger traffic per year
avg_traffic_per_year = df[year_columns].mean()

# Find the year with the highest average
max_avg_year = avg_traffic_per_year.idxmax()
max_avg_value = avg_traffic_per_year.max()

# Calculate standard deviation for that year
std_dev = df[max_avg_year].std()

print(f"Final Answer: {max_avg_year}, {std_dev:.0f}")