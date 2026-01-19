import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% of party list votes' to float and find the max
df['% of party list votes'] = df['% of party list votes'].str.replace(',', '').astype(float)
max_percentage_row = df.loc[df['% of party list votes'].idxmax()]
final_year = max_percentage_row['Election year']
print(f"Final Answer: {final_year}")