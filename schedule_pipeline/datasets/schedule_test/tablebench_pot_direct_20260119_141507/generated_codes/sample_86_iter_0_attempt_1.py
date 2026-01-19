import pandas as pd

df = pd.read_csv('table.csv')
# Filter row where nation is 'switzerland' and sum all medal types
switzerland_data = df[df['nation'] == 'switzerland']
total_medals = switzerland_data['gold'].values[0] + switzerland_data['silver'].values[0] + switzerland_data['bronze'].values[0]
print(f"Final Answer: {total_medals}")