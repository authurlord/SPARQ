import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'Transparency International'
filtered_df = df[df['organization'] == 'Transparency International']
# Find the year with the lowest rank (highest rank)
best_year = filtered_df.loc[filtered_df['rank'].idxmin(), 'year']
print(f"Final Answer: {best_year}")