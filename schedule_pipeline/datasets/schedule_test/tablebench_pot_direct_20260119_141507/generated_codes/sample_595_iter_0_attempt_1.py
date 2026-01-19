import pandas as pd

df = pd.read_csv('table.csv')
# Filter counties where Obama received more than 60%
filtered_df = df[df['obama%'].str.contains('%', na=False) & (df['obama%'].str.rstrip('%').astype(float) > 60)]
# Sum the 'total' column for those filtered rows
total_votes = filtered_df['total'].sum()
print(f"Final Answer: {total_votes}")