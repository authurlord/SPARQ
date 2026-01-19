import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'average' column and find the max and min values
top_average = df.loc[df['rank by average'] == '1', 'average'].values[0]
bottom_average = df.loc[df['rank by average'] == '12', 'average'].values[0]
difference = top_average - bottom_average
print(f"Final Answer: {difference:.1f}")