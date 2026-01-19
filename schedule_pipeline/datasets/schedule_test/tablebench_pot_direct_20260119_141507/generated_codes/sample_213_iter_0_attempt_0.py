import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between lane and time
correlation = df['lane'].corr(df['time'])
print(f"Final Answer: {correlation:.3f}")