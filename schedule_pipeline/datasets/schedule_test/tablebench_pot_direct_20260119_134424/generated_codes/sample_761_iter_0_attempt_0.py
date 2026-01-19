import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the average number of bills originally cosponsored
forecast = df['bills originally cosponsored'].mean()
print(f"Final Answer: {forecast:.1f}")