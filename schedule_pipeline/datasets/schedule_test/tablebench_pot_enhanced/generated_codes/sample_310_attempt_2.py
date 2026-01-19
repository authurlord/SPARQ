import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Emmy Awards for a TV series
emmy_series_wins = df[(df['Association'] == 'Emmy Awards') & 
                      (df['Category'].str.contains('Series|TV', na=False)) & 
                      (df['Result'] == 'Won')]
# Find the year she won for a TV series
year = emmy_series_wins['Year'].iloc[0] if not emmy_series_wins.empty else None
print(f"Final Answer: {year}")