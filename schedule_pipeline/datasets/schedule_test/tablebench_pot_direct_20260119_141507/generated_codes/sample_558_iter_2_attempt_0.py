import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer for proper filtering
df['Year'] = df['Year'].astype(int)
# Filter rows from 1935 to 1943 inclusive
filtered_df = df[(df['Year'] >= 1935) & (df['Year'] <= 1943)]
# Calculate the average number of locomotives withdrawn
average_withdrawn = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {average_withdrawn:.1f}")