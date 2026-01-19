import pandas as pd

df = pd.read_csv('table.csv')
# Filter row where nation is 'switzerland' and sum all medal types
switzerland_medals = df[df['nation'] == 'switzerland']['gold'].iloc[0] + df[df['nation'] == 'switzerland']['silver'].iloc[0] + df[df['nation'] == 'switzerland']['bronze'].iloc[0]
print(f"Final Answer: {switzerland_medals}")