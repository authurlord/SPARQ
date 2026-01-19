import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years from 1935 to 1943
filtered_df = df[(df['Year'] >= '1935') & (df['Year'] <= '1943')]
# Calculate the average of 'Quantity withdrawn'
avg_withdrawn = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {avg_withdrawn:.1f}")