import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum crude birth rate
max_birth_rate_row = df.loc[df['Crude birth rate (per 1000)'].idxmax()]
year_with_highest_birth_rate = max_birth_rate_row['Unnamed: 0']
print(f"Final Answer: {year_with_highest_birth_rate}")