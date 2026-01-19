import pandas as pd

df = pd.read_csv('table.csv')
# Filter only ambassadors
ambassadors = df[df['Title'] == 'Ambassador']
# Convert Start and End to integers
ambassadors['Start'] = pd.to_numeric(ambassadors['Start'])
ambassadors['End'] = pd.to_numeric(ambassadors['End'])
# Calculate years served (inclusive)
years_served = (ambassadors['End'] - ambassadors['Start'] + 1).sum()
print(f"Final Answer: {years_served}")