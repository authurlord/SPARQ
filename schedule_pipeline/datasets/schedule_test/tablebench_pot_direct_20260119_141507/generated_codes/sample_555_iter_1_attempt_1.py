import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'Transparency International'
transparency_rows = df[df['organization'] == 'Transparency International']

# If any rows exist, find the one with the minimum rank (highest rank value)
if not transparency_rows.empty:
    min_rank_row = transparency_rows.loc[transparency_rows['rank'].idxmin()]
    final_year = min_rank_row['year']
    print(f"Final Answer: {final_year}")
else:
    print("Final Answer: None")