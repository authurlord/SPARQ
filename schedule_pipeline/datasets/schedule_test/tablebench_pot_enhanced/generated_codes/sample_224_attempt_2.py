import pandas as pd

df = pd.read_csv('table.csv')
# Sort by projectile diameter to observe the trend
sorted_df = df.sort_values(by='p1 diameter (mm)')
print(sorted_df[['p1 diameter (mm)', 'p max ( bar )']])