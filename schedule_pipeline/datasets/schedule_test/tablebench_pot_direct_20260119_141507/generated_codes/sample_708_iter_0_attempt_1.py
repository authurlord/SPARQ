import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the country has "(2001)" in it
filtered_df = df[df['country'].str.contains(r'\(2001\)', na=False)]
# Extract the country name (before the parentheses) and convert percentage to float
filtered_df['orphans as % of all children'] = filtered_df['orphans as % of all children'].str.replace('%', '').astype(float)
# Find the row with the maximum percentage
max_row = filtered_df.loc[filtered_df['orphans as % of all children'].idxmax()]
country_with_highest_percentage = max_row['country'].split(' (')[0]
print(f"Final Answer: {country_with_highest_percentage}")