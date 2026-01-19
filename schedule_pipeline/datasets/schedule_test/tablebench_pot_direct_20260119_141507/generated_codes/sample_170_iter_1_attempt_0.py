import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Natural change (per 1000)' to numeric (some entries have '- 27' etc.)
df['Natural change (per 1000)'] = df['Natural change (per 1000)'].str.replace(' - ', '-').astype(float)

# Find the year with the highest natural change
max_change_row = df.loc[df['Natural change (per 1000)'].idxmax()]
max_year = max_change_row['Unnamed: 0']

# Also check for a significant jump (e.g., from negative to positive)
df['Natural change (per 1000)'] = pd.to_numeric(df['Natural change (per 1000)'], errors='coerce')
df['change_diff'] = df['Natural change (per 1000)'].diff()

# Find years where change_diff is large positive (jump up)
jump_years = df[(df['change_diff'] > 2) & (df['Natural change (per 1000)'] > 0)]['Unnamed: 0']
jump_years_list = jump_years.tolist()

# Also include the peak year
peak_year = df.loc[df['Natural change (per 1000)'].idxmax()]['Unnamed: 0']

# If jump years exist, combine; otherwise, just peak
if len(jump_years_list) > 0:
    significant_years = jump_years_list + [peak_year]
else:
    significant_years = [peak_year]

# Remove duplicates and sort
significant_years = sorted(list(set(significant_years)))

print(f"Final Answer: {significant_years[0]}, {significant_years[1]}")