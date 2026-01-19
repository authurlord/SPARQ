import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns
columns_to_analyze = ['candidates', 'average votes per candidate', '% of total vote', 'average % of vote per candidate', 'saved deposits']
correlation_with_total_votes = df[columns_to_analyze].corrwith(df['total votes'])

# Find the factor with the highest absolute correlation
main_factor = correlation_with_total_votes.abs().idxmax()

print(f"Final Answer: {main_factor}")