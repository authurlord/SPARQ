import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'obama%' to float by removing the % sign and dividing by 100
df['obama%'] = df['obama%'].str.rstrip('%').astype(float)
# Filter rows where Obama's percentage is greater than 60
filtered_df = df[df['obama%'] > 60]
# Sum the total votes in these counties
total_votes = filtered_df['total'].sum()
print(f"Final Answer: {total_votes}")