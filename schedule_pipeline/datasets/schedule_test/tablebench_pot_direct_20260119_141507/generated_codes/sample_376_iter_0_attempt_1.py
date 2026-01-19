import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of companies in the banking industry
banking_count = df[df['industry'] == 'banking'].shape[0]
print(f"Final Answer: {banking_count}")