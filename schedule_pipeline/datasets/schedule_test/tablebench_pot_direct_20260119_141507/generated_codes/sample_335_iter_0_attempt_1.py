import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is 'Heptathlon' and Competition is 'South American Championships' and Venue is 'Lima, Peru'
filtered_df = df[(df['Event'] == 'Heptathlon') & (df['Competition'] == 'South American Championships') & (df['Venue'] == 'Lima, Peru')]
# Extract the year and points score
year = filtered_df.iloc[0]['Year']
points = filtered_df.iloc[0]['Notes']
print(f"Final Answer: {year}, {points}")