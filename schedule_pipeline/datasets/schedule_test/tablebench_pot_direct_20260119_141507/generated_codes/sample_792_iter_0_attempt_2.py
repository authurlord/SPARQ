import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows from 1975 to 1990 inclusive
filtered_df = df[(df['Year'] >= 1975) & (df['Year'] <= 1990)]
# Calculate standard deviation of 'Political Rights'
std_political_rights = filtered_df['Political Rights'].std()
print(f"Final Answer: {std_political_rights:.2f}")