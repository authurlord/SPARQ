import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'Magnitude' and 'Depth (km)'
correlation = df['Magnitude'].corr(df['Depth (km)'])
print(f"Final Answer: {correlation:.3f}")