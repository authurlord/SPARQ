import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'year' column to integer for proper filtering
df['year'] = df['year'].astype(int)
# Filter rows where year is between 2002 and 2005 inclusive
filtered_mintage = df[(df['year'] >= 2002) & (df['year'] <= 2005)]['mintage'].sum()
# Divide by 5 to get coins per person
coins_per_person = filtered_mintage / 5
print(f"Final Answer: {coins_per_person}")