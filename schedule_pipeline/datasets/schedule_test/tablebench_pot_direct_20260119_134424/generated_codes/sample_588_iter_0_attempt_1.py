import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 1965
row_1965 = df[df['election'] == '1965']
# Extract the national votes and convert to integer
national_votes_1965 = int(row_1965['of national votes'].values[0])
# Increase by 10%
new_national_votes = national_votes_1965 * 1.1
print(f"Final Answer: {new_national_votes:.0f}")