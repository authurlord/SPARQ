import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to float by removing '%' and converting to numeric
df['success rate'] = df['success rate'].str.rstrip('%').astype(float)
# Find the year with the maximum success rate
max_rate_row = df.loc[df['success rate'].idxmax()]
highest_success_year = max_rate_row['year']
print(f"Final Answer: {highest_success_year}")