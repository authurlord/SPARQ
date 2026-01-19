import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'pr seats' column to get total PR seats won by all parties
total_pr_seats = df['pr seats'].sum()
print(f"Final Answer: {total_pr_seats}")