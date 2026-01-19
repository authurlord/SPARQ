import pandas as pd

df = pd.read_csv('table.csv')
# Filter only ambassadors
ambassadors = df[df['Title'] == 'Ambassador']
# Convert Start and End to integers
ambassadors['Start'] = ambassadors['Start'].astype(int)
ambassadors['End'] = ambassadors['End'].astype(int)
# Calculate years served for each ambassador
ambassadors['Years'] = ambassadors['End'] - ambassadors['Start'] + 1
# Sum total years served
total_years = ambassadors['Years'].sum()
print(f"Final Answer: {total_years}")