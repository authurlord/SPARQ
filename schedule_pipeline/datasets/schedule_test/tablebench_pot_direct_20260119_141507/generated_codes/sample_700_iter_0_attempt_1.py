import pandas as pd

df = pd.read_csv('table.csv')
# Remove the '%' and convert success rate to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)
# Find the year with the maximum success rate
max_success_year = df.loc[df['success rate'].idxmax(), 'year']
print(f"Final Answer: {max_success_year}")