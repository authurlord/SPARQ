import pandas as pd

df = pd.read_csv('table.csv')
# Convert success rate to numeric by removing % and converting to float
df['success rate'] = df['success rate'].str.replace('%', '').astype(float)
# Find the year with the highest success rate (excluding the 'total' row)
max_success_year = df.loc[df['success rate'].idxmax()]['year']
print(f"Final Answer: {max_success_year}")