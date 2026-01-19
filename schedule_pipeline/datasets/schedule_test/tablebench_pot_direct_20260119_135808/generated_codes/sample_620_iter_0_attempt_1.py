import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 2002 to 2005
filtered_df = df[(df['year'].astype(int) >= 2002) & (df['year'].astype(int) <= 2005)]
# Calculate total mintage
total_mintage = filtered_df['mintage'].astype(int).sum()
# Divide equally among 5 people
coins_per_person = total_mintage / 5
print(f"Final Answer: {int(coins_per_person)}")