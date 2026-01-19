import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where wins is 13
winning_season = df[df['wins'] == 13]['season'].iloc[0]
print(f"Final Answer: {winning_season}")