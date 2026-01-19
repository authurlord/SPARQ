import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains in Tanzania with elevation > 3000 m and prominence < 3000 m
filtered_mountains = df[
    (df['country'] == 'tanzania') &
    (df['elevation (m)'] > 3000) &
    (df['prominence (m)'] < 3000)
]
count = len(filtered_mountains)
print(f"Final Answer: {count}")