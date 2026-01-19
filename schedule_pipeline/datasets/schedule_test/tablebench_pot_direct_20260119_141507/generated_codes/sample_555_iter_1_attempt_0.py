import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'Transparency International'
transparency_rows = df[df['organization'] == 'Transparency International']
# If any rows exist, find the one with minimum rank (best rank)
if not transparency_rows.empty:
    best_year = transparency_rows.loc[transparency_rows['rank'].idxmin(), 'year']
    print(f"Final Answer: {best_year}")
else:
    print("Final Answer: None")