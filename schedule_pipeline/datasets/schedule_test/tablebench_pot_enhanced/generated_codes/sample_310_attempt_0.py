import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Emmy Awards and TV series-related work
emmy_series_win = df[(df['Association'] == 'Emmy Awards') & 
                     (df['Category'].str.contains('Series', na=False)) & 
                     (df['Result'] == 'Won')]

# Extract the year of the win
year = emmy_series_win['Year'].iloc[0] if not emmy_series_win.empty else None
print(f"Final Answer: {year}")