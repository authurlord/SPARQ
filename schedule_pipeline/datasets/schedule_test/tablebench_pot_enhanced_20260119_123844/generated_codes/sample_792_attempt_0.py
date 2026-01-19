import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years from 1975 to 1990
filtered_df = df[(df['Year'] >= '1975') & (df['Year'] <= '1990')]
# Calculate standard deviation of 'Political Rights'
std_political_rights = filtered_df['Political Rights'].astype(int).std()
print(f"Final Answer: {std_political_rights:.2f}")