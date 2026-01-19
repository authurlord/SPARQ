import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Year is between 1975 and 1990 (inclusive)
filtered_df = df[(df['Year'] >= 1975) & (df['Year'] <= 1990)]
# Calculate standard deviation of 'Political Rights' in the filtered data
std_political_rights = filtered_df['Political Rights'].std()
print(f"Final Answer: {std_political_rights:.2f}")