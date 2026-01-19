import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Grammy Awards where the result is 'Won' and the nominated work is 'Love Song'
filtered_df = df[(df['Association'] == 'Grammy Awards') & 
                  (df['Result'] == 'Won') & 
                  (df['Nominated work'] == 'Love Song')]
# Extract the year
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")