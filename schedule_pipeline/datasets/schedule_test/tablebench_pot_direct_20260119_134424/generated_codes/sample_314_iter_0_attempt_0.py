import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Association is 'Grammy Awards', Category is 'Song of the Year', and Result is 'Won'
winning_year = df[(df['Association'] == 'Grammy Awards') & 
                  (df['Category'] == 'Song of the Year') & 
                  (df['Result'] == 'Won')]['Year'].iloc[0]
print(f"Final Answer: {winning_year}")