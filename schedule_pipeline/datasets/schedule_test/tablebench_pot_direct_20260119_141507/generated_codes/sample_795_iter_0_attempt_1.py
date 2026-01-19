import pandas as pd

df = pd.read_csv('table.csv')

# Convert the traffic columns to numeric
traffic_cols = ['2008', '2009', '2010']
df[traffic_cols] = df[traffic_cols].apply(pd.to_numeric, errors='coerce')

# Calculate average passenger traffic per year
annual_avg = df[traffic_cols].mean(axis=0)

# Find the year with the highest average
max_year = annual_avg.idxmax()
max_avg = annual_avg[max_year]

# Calculate standard deviation for that year
std_dev = df[max_year].std()

print(f"Final Answer: {max_year}, {std_dev:.1f}")