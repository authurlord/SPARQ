import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Emmy Awards for a TV series and specifically for 'The Barbara Stanwyck Show'
emmy_series_win = df[(df['Association'] == 'Emmy Awards') & 
                     (df['Category'].str.contains('Series', na=False)) & 
                     (df['Work'] == 'The Barbara Stanwyck Show') & 
                     (df['Result'] == 'Won')]

# Get the year
year = emmy_series_win['Year'].iloc[0]
print(f"Final Answer: {year}")