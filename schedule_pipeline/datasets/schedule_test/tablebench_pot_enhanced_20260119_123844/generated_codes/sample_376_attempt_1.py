import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'banking'
banking_companies = df[df['industry'] == 'banking']
# Count the number of banking companies
num_bank_companies = len(banking_companies)
print(f"Final Answer: {num_bank_companies}")