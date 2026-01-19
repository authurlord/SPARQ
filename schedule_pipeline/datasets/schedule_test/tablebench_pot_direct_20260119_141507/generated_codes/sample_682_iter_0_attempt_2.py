import pandas as pd

df = pd.read_csv('table.csv')
# Convert string values to float
df['1990 - 95'] = pd.to_numeric(df['1990 - 95'], errors='coerce')
df['2006 - 10'] = pd.to_numeric(df['2006 - 10'], errors='coerce')

# Compute correlation between the two columns
correlation = df['1990 - 95'].corr(df['2006 - 10'])
print(f"Final Answer: {correlation:.3f}")