import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats won' to numeric, coercing errors to NaN (though all should be valid)
df['seats won'] = pd.to_numeric(df['seats won'], errors='coerce')

# Count parties where seats won is 10 or fewer
count_parties = (df['seats won'] <= 10).sum()
print(f"Final Answer: {count_parties}")