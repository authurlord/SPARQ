import pandas as pd

df = pd.read_csv('table.csv')

# Extract points per year
points_per_year = df[df['year'] != 'total']['points'].astype(int)

# Since there's no data or model for future predictions, we cannot forecast.
print("Final Answer: Cannot predict")