import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the data to a DataFrame with proper column names
# The first row is the header, and the rest are data
# The 'total' row is the sum of all counties; we exclude it since we're interested in counties
# Extract only the county rows (excluding the last row 'total')
county_data = df.iloc[:-1]  # Exclude the total row

# Convert the columns to numeric (remove commas if any, though not present here)
county_data = county_data.apply(pd.to_numeric, errors='coerce')

# Compute year-over-year percentage change for each county
# Columns are years: 1948, 1956, 1966, 1977, 1992, 2002, 2011
# We need to calculate the change between consecutive years
years = ['1948', '1956', '1966', '1977', '1992', '2002', '2011']
changes = []

for idx, row in county_data.iterrows():
    # Extract values for each year
    values = row[years]
    # Drop NaNs
    clean_values = values.dropna()
    # If less than 2 values, skip
    if len(clean_values) < 2:
        continue
    # Calculate percentage change between consecutive years
    changes_in_row = []
    for i in range(1, len(clean_values)):
        prev_val = clean_values[i-1]
        curr_val = clean_values[i]
        if prev_val == 0:
            continue
        pct_change = ((curr_val - prev_val) / prev_val) * 100
        changes_in_row.append(abs(pct_change))
    # If any change exceeds 20%, mark this county
    if any(change > 20 for change in changes_in_row):
        changes.append(row['county'])

# Get unique counties with unusual patterns
unusual_counties = list(set(changes))

# If no unusual patterns found, return empty list
if not unusual_counties:
    unusual_counties = []

print(f"Final Answer: {', '.join(unusual_counties)}")