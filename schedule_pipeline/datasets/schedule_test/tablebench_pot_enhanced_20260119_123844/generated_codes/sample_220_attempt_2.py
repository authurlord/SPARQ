import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 1959 and 1989 (inclusive)
filtered_df = df[(df['year (january)'] >= '1959') & (df['year (january)'] <= '1989')]
# Display the urban percentage and population for these years
print("Urban percentage vs Population (1959-1989):")
for index, row in filtered_df.iterrows():
    print(f"Year: {row['year (january)']}, Population: {row['population (000)']}k, Urban %: {row['urban , %']}")