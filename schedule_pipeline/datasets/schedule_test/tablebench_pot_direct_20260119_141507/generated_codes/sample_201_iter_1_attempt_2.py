import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out rows that are headers or totals (e.g., 'Career Total', 'Liverpool Total')
# We focus on regular season data, excluding summary rows
season_rows = df[df['Club'].str.contains('Real Madrid|Liverpool|Manchester City') == False]
# Actually, keep only the rows where Club is not a total row
filtered_df = df[df['Club'] != 'Career Total']
filtered_df = filtered_df[~filtered_df['Club'].str.contains('Total', case=False)]

# Select only 'Apps' and 'Goals' columns
app_goals_data = filtered_df[['Apps', 'Goals']]

# Calculate the correlation coefficient
correlation = app_goals_data['Apps'].corr(app_goals_data['Goals'])

print(f"Final Answer: {correlation:.2f}")