import pandas as pd

df = pd.read_csv('table.csv')
# Find the team in 3rd position
third_place_team = df[df['position'] == '3']['team'].values[0]
print(f"Final Answer: {third_place_team}")