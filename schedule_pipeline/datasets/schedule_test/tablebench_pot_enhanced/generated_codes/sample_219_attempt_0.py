import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for calculation
df['total usaaf'] = pd.to_numeric(df['total usaaf'])
df['overseas'] = pd.to_numeric(df['overseas'])

# Calculate correlation coefficient
correlation = df['total usaaf'].corr(df['overseas'])

# Check if correlation is positive (indicating increase together)
if correlation > 0:
    print("Final Answer: yes")
else:
    print("Final Answer: no")