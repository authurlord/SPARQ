import pandas as pd

df = pd.read_csv('table.csv')
# Calculate growth rate from 2008 to 2009
df['growth_rate'] = ((df['2009'] - df['2008']) / df['2008']) * 100
# Find the airport with the maximum growth rate
max_growth_row = df.loc[df['growth_rate'].idxmax()]
final_airport = max_growth_row['airport']
print(f"Final Answer: {final_airport}")