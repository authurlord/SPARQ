import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2002 to 2005
filtered_df = df[(df['year'].between(2002, 2005))]
# Convert mintage to integer and sum
total_mintage = filtered_df['mintage'].str.replace(',', '').astype(int).sum()
# Divide by 5
coins_per_person = total_mintage // 5
print(f"Final Answer: {coins_per_person}")