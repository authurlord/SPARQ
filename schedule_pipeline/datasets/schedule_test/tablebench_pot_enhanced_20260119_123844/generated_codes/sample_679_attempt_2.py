import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between the two columns
correlation = df['carbon dioxide emissions per year (tons per person) (2007)'].corr(df['avg emission per km 2 of its land (tons)'])
print(f"Final Answer: {correlation:.4f}")