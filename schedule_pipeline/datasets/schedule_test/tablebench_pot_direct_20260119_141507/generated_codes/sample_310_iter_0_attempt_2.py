import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Work is 'The Barbara Stanwyck Show' and Result is 'Won'
barbara_stanwyck_emmy = df[(df['Work'] == 'The Barbara Stanwyck Show') & (df['Result'] == 'Won')]
# Extract the year
year = barbara_stanwyck_emmy['Year'].values[0]
print(f"Final Answer: {year}")