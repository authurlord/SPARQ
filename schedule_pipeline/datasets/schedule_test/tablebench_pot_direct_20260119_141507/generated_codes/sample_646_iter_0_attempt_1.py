import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'Average population (x 1000)' and 'Natural change (per 1000)'
correlation = df['Average population (x 1000)'].corr(df['Natural change (per 1000)'])
print(f"Final Answer: {correlation:.3f}")