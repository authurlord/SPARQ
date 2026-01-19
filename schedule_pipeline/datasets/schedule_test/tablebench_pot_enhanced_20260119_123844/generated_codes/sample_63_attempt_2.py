import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Lionel Palairet
lionel_palairet_row = df[df['player'] == 'lionel palairet']
# Extract the highest score
highest_score = lionel_palairet_row['highest score'].values[0]
print(f"Final Answer: {highest_score}")