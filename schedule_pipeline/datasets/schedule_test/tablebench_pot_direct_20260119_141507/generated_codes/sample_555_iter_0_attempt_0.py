import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'Transparency International'
ti_rows = df[df['organization'] == 'Transparency International']
# Find the row with the lowest rank (highest-ranked index)
best_rank_row = ti_rows.loc[ti_rows['rank'].idxmin()]
# Extract the year
year = best_rank_row['year']
print(f"Final Answer: {year}")