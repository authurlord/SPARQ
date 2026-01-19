import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where 'country / territory' is 'australia'
capital = df[df['country / territory'] == 'australia']['capital'].iloc[0]
print(f"Final Answer: {capital}")