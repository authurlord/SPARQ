import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between elevation and prominence
correlation = df['elevation (m)'].corr(df['prominence (m)'])
print(f"Final Answer: {correlation:.3f}")