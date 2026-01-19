import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is 'Heptathlon' and Venue is 'Lima, Peru'
filtered_df = df[(df['Event'] == 'Heptathlon') & (df['Venue'] == 'Lima, Peru')]
# Extract the year and points (from Notes column)
result_year = filtered_df.iloc[0]['Year']
result_points = filtered_df.iloc[0]['Notes']
print(f"Final Answer: {result_year}, {result_points}")