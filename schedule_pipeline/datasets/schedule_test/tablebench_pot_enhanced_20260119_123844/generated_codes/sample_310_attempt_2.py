import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Emmy Awards for a TV series where she won
emmy_series_win = df[(df['Association'] == 'Emmy Awards') & 
                     (df['Category'].str.contains('Series', na=False)) & 
                     (df['Result'] == 'Won')]

# Extract the year
year = emmy_series_win['Year'].iloc[0]
print(f"Final Answer: {year}")