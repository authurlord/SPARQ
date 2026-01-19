import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert '% of party list votes' to float, handling None and comma-separated values
df['% of party list votes'] = df['% of party list votes'].astype(str).str.replace(',', '').astype(float)
# Find the election year with the maximum '% of party list votes'
max_year = df.loc[df['% of party list votes'].idxmax(), 'Election year']
print(f"Final Answer: {max_year}")