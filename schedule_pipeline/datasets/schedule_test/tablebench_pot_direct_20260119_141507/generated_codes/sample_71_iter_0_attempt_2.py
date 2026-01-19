import pandas as pd

df = pd.read_csv('table.csv')
# Find the capital of Australia
australia_capital = df[df['country / territory'] == 'australia']['capital'].values[0]
print(f"Final Answer: {australia_capital}")