import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the mean of 'population density (/ km²)'
avg_density = df['population density ( / km 2 )'].mean()
print(f"Final Answer: {avg_density:.2f}")