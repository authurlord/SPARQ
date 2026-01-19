import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2001 - 05' column to float for numerical comparison
df['2001 - 05'] = pd.to_numeric(df['2001 - 05'])
# Count states with value > 0.3 in '2001 - 05'
count = (df['2001 - 05'] > 0.3).sum()
print(f"Final Answer: {count}")