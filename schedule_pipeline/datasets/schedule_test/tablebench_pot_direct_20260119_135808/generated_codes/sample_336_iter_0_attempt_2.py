import pandas as pd

df = pd.read_csv('table.csv')

# Filter out invalid entries
valid_rows = df[
    (df['senior pga championship'] != 'not held') &
    (df['senior pga championship'] != 'founded in') &
    (df['senior players championship'] != 'not held') &
    (df['senior players championship'] != 'founded in')
]

# Find rows where the same athlete won both titles
win_both = valid_rows[
    valid_rows['senior pga championship'] == valid_rows['senior players championship']
]

# Extract the year
if not win_both.empty:
    year = win_both['year'].iloc[0]
else:
    year = "No such year found"

print(f"Final Answer: {year}")