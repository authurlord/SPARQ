import pandas as pd

df = pd.read_csv('table.csv')
# Convert mintage to integer and filter years 2002 to 2005
mintage_values = df[(df['year'].between(2002, 2005))]['mintage'].astype(int)
total_mintage = mintage_values.sum()
coins_per_person = total_mintage / 5
print(f"Final Answer: {coins_per_person}")