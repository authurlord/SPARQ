import pandas as pd

df = pd.read_csv('table.csv')
# Get average scores for top and bottom ranked couples
top_avg = float(df.loc[df['rank by average'] == '1', 'average'].values[0])
bottom_avg = float(df.loc[df['rank by average'] == '12', 'average'].values[0])
difference = top_avg - bottom_avg
print(f"Final Answer: {difference:.1f}")