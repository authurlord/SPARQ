import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the values for years between 1850 and 1870
# The values are in the second column (index 1), which has values like '8,385'
values = df[df['Year'].between(1850, 1870)]['-'].str.replace(',', '').astype(int)

# Extract the years corresponding to these values
years = df[df['Year'].between(1850, 1870)]['Year'].tolist()

# Compute the increases between consecutive years
increases = []
for i in range(len(years) - 1):
    increase = values.iloc[i+1] - values.iloc[i]
    increases.append(increase)

# Find the maximum increase and the year when it occurs
max_increase = max(increases)
# The year where the increase happens is the year before the one with the max increase
# So we find the index of the max increase in the increases list
max_increase_index = increases.index(max_increase)
year_with_max_increase = years[max_increase_index + 1]

print(f"Final Answer: {year_with_max_increase}")