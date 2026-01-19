import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Obama's percentage is greater than 60%
filtered_df = df[df['obama%'].str.replace('%', '').astype(float) > 60]
# Sum the total votes in these counties
total_votes = filtered_df['total'].sum()
print(f"Final Answer: {total_votes}")