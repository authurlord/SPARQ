import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Emmy Awards for a TV series where she won
filtered_df = df[(df['Association'] == 'Emmy Awards') & 
                 (df['Category'].str.contains('Series', na=False)) & 
                 (df['Result'] == 'Won')]
# Get the year
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")