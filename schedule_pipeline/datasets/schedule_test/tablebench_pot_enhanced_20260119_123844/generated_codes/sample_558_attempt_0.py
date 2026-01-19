import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1935 to 1943
filtered_df = df[(df['Year'].astype(int) >= 1935) & (df['Year'].astype(int) <= 1943)]
# Calculate the average of 'Quantity withdrawn'
average_withdrawn = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {average_withdrawn:.1f}")