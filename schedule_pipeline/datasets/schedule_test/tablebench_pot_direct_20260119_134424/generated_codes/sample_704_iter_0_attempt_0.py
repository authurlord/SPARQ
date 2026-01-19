import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of party list votes' to numeric, coercing errors to NaN
df['% of party list votes'] = pd.to_numeric(df['% of party list votes'], errors='coerce')
# Find the election year with the maximum '% of party list votes'
max_year = df.loc[df['% of party list votes'].idxmax(), 'Election year']
print(f"Final Answer: {max_year}")