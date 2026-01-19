import pandas as pd

df = pd.read_csv('table.csv')
# Convert the year columns to numeric
year_columns = ['2008', '2009', '2010']
df[year_columns] = df[year_columns].apply(pd.to_numeric)

# Calculate average passenger traffic per year
avg_traffic = df[year_columns].mean()

# Find the year with the highest average
highest_avg_year = avg_traffic.idxmax()
highest_avg_value = avg_traffic.max()

# Calculate standard deviation for that year
std_dev = df[highest_avg_year].std()

print(f"Final Answer: {highest_avg_year}, {std_dev:.1f}")