import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter rows where City is 'Moscow' and Total is not missing
moscow_data = df[(df['City'] == 'Moscow') & (df['Number of Examinees by Level_4'].notna())]

# Convert Year to integer and sort by Year
moscow_data['Year'] = pd.to_numeric(moscow_data['Year'], errors='coerce')
moscow_data = moscow_data.dropna(subset=['Year'])
moscow_data = moscow_data.sort_values('Year')

# Extract the Total column (last column)
totals = moscow_data['Number of Examinees by Level_4'].values

# Compute year-over-year percentage increase
years = moscow_data['Year'].values
percentage_increases = []

for i in range(1, len(years)):
    prev_total = totals[i-1]
    curr_total = totals[i]
    if prev_total == 0:
        continue
    increase = ((curr_total - prev_total) / prev_total) * 100
    percentage_increases.append(increase)

# Find the year with the maximum increase
max_increase_idx = percentage_increases.index(max(percentage_increases)) if percentage_increases else None
max_increase_year = years[max_increase_idx + 1] if max_increase_idx is not None else None

print(f"Final Answer: {max_increase_year}")