import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Convert to a dictionary for easier comparison
eu_totals = dict(zip(eu_data['year'], eu_data['total'].astype(float)))
us_totals = dict(zip(us_data['year'], us_data['total'].astype(float)))

# Find the year when EU total exceeds U.S. total
for year in sorted(set(eu_totals.keys()) | set(us_totals.keys())):
    eu_total = eu_totals.get(year, 0)
    us_total = us_totals.get(year, 0)
    if eu_total > us_total:
        print(f"Final Answer: {year}")
        break