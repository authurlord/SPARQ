import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for the year 1995
crude_birth_rate_1995 = df[df['Unnamed: 0'] == '1995']['Crude birth rate (per 1000)'].values[0]
print(f"Final Answer: {crude_birth_rate_1995}")