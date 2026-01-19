import pandas as pd

df = pd.read_csv('table.csv')
# Convert the years to numeric
years = ['2008', '2009', '2010']
df[years] = df[years].apply(pd.to_numeric)

# Calculate the average passenger traffic per year
avg_traffic = df[years].mean(axis=0)
max_avg_year = avg_traffic.idxmax()

# Calculate standard deviation for the year with highest average
std_dev = df[max_avg_year].std()

print(f"Final Answer: {max_avg_year}, {std_dev:.2f}")