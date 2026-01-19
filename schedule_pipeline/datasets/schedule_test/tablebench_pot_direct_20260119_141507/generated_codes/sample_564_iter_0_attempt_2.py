import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years between 2004 and 2013 inclusive
filtered_df = df[(df['Year'].astype(int) >= 2004) & (df['Year'].astype(int) <= 2013)]
# Calculate the mean of 'No. of performances'
average_performances = filtered_df['No. of performances'].mean()
print(f"Final Answer: {average_performances:.1f}")