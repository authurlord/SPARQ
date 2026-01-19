import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years 2002 to 2005
filtered_df = df[(df['year'] >= 2002) & (df['year'] <= 2005)]
# Sum the mintage and divide by 5
total_mintage = filtered_df['mintage'].sum()
coins_per_person = total_mintage / 5
print(f"Final Answer: {coins_per_person}")