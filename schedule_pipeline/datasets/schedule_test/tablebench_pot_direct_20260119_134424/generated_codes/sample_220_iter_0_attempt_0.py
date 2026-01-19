import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 1959 and 1989
filtered_df = df[(df['year (january)'] >= '1959') & (df['year (january)'] <= '1989')]
# Display the relevant columns to observe the trend
print("Urban percentage vs Population (1959-1989):")
print(filtered_df[['year (january)', 'population (000)', 'urban , %']])