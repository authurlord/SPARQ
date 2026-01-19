import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the industry is 'banking'
banking_companies = df[df['industry'] == 'banking']
# Count the number of banking companies
count_bankings = len(banking_companies)
print(f"Final Answer: {count_bankings}")