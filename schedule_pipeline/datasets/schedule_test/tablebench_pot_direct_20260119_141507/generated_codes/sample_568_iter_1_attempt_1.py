import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for European Union and United States
eu_data = df[df['country'] == 'european union']
us_data = df[df['country'] == 'united states']

# Extract years and total values
eu_years = eu_data['year'].dropna().tolist()
us_years = us_data['year'].dropna().tolist()

# Find intersection of years
common_years = set(eu_years) & set(us_years)

# If there is a common year, check if EU total > US total
if common_years:
    # Filter data for common years
    common_df = df[df['year'].isin(common_years)]
    # Sort by year
    common_df = common_df.sort_values('year')
    # Check if EU total exceeds US total in any year
    for _, row in common_df.iterrows():
        if row['country'] == 'european union' and row['total'] > row['total']:
            # This won't happen because it's same row
            pass
    # Actually, we need to compare EU and US separately
    # We'll create a new DataFrame with both
    eu_us_comparison = df[df['country'].isin(['european union', 'united states'])]
    eu_us_comparison = eu_us_comparison.sort_values('year')
    
    # Group by year and compare
    year_comparison = {}
    for year in sorted(eu_us_comparison['year'].unique()):
        eu_row = eu_us_comparison[(eu_us_comparison['country'] == 'european union') & (eu_us_comparison['year'] == year)]
        us_row = eu_us_comparison[(eu_us_comparison['country'] == 'united states') & (eu_us_comparison['year'] == year)]
        if len(eu_row) > 0 and len(us_row) > 0:
            eu_total = float(eu_row['total'].iloc[0])
            us_total = float(us_row['total'].iloc[0])
            if eu_total > us_total:
                year_comparison[year] = True
    if year_comparison:
        final_year = list(year_comparison.keys())[0]
        print(f"Final Answer: {final_year}")
    else:
        print("Final Answer: none")
else:
    print("Final Answer: none")