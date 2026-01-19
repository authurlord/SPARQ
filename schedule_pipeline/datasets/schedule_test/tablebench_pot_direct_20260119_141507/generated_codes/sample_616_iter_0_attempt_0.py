import pandas as pd

df = pd.read_csv('table.csv')
# Extract numeric values from 'hk viewers' column by removing 'million' and converting to float
df['hk viewers'] = df['hk viewers'].str.replace(' million', '').astype(float)
# Sum the average viewership for the top 5 shows (first 5 rows)
total_viewership = df.head(5)['hk viewers'].sum()
print(f"Final Answer: {total_viewership:.2f}")