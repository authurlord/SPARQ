import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where position is 3
third_position_team = df[df['position'] == '3']['team'].values[0]
print(f"Final Answer: {third_position_team}")