import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Remove the 'total' row
df = df[df['county'] != 'total']

# Convert the numeric columns to integers
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate year-on-year differences
years = ['1948', '1956', '1966', '1977', '1992', '2002', '2011']
df_years = df.copy()
df_years.set_index('county', inplace=True)

# Compute year-on-year differences
diffs = {}
for i in range(1, len(years)):
    year1, year2 = years[i-1], years[i]
    diff_col = f"{year2} - {year1}"
    df_years[diff_col] = df_years[year2] - df_years[year1]

# Identify counties with large changes (absolute difference > 100,000)
threshold = 100000
unusual_counties = []

for county in df_years.index:
    row_diffs = df_years.loc[county, [f"{y} - {y2}" for y2 in years[1:] for y in years[:len(years)-1]]]
    # Check if any year-on-year change exceeds threshold
    if any(abs(row_diffs[d]) > threshold for d in row_diffs):
        unusual_counties.append(county)

# Final answer: list of counties with unusual patterns
print(f"Final Answer: {', '.join(unusual_counties)}")