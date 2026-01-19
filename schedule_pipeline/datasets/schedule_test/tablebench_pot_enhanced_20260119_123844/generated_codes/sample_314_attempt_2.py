import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Category is 'Song of the Year' and Result is 'Won'
winner_row = df[(df['Category'] == 'Song of the Year') & (df['Result'] == 'Won')]
# Extract the year
year = winner_row['Year'].iloc[0]
print(f"Final Answer: {year}")