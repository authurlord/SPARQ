import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Switzerland
switzerland_row = df[df['nation'] == 'switzerland']
# Calculate total medals
total_medals = switzerland_row['gold'].values[0] + switzerland_row['silver'].values[0] + switzerland_row['bronze'].values[0]
print(f"Final Answer: {total_medals}")