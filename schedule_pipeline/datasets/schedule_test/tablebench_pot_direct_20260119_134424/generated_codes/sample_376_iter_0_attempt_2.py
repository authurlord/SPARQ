import pandas as pd

df = pd.read_csv('table.csv')
# Filter the rows where 'industry' is 'banking'
banking_companies = df[df['industry'] == 'banking']
# Count the number of banking companies
count_bankings = len(banking_companies)
print(f"Final Answer: {count_bankings}")