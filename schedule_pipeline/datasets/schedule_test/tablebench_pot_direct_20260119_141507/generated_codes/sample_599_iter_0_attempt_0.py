import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is 1944
filtered_df = df[df['Year'] == '1944']
# Calculate the mean of 'US Chart position'
average_position = filtered_df['US Chart position'].mean()
print(f"Final Answer: {average_position:.1f}")