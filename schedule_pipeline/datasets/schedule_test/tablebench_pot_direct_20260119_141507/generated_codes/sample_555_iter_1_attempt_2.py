import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'Transparency International'
transparency_rows = df[df['organization'] == 'transparency international']
# If any rows exist, find the one with the minimum rank (highest rank = lowest number)
if not transparency_rows.empty:
    min_rank_row = transparency_rows.loc[transparency_rows['rank'].idxmin()]
    year = min_rank_row['year']
    print(f"Final Answer: {year}")
else:
    print("Final Answer: None")