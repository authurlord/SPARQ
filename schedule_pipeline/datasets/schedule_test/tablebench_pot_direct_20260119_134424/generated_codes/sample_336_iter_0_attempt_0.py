import pandas as pd

df = pd.read_csv('table.csv')

# Filter out invalid entries
valid_rows = df[
    (df['senior pga championship'] != 'not held') &
    (df['senior pga championship'] != 'founded in') &
    (df['senior players championship'] != 'not held') &
    (df['senior players championship'] != 'founded in')
]

# Check for rows where the same athlete won both titles
matching_rows = valid_rows[
    valid_rows['senior pga championship'] == valid_rows['senior players championship']
]

# Extract the year
if not matching_rows.empty:
    year = matching_rows['year'].values[0]
    print(f"Final Answer: {year}")
else:
    print("Final Answer: None")