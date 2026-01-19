import pandas as pd

df = pd.read_csv('table.csv')

# Describe the table
print("Table Features:")
print("The table contains data on commemorative coins issued between 2003 and 2005.")
print("Columns:")
print(" - year: Year of issue.")
print(" - design: Theme of the coin.")
print(" - issue: Special feature (e.g., hologram, gold plated).")
print(" - artist: Designer of the coin.")
print(" - mintage: Number of coins minted.")
print(" - issue price: Price at issuance.")

print("\nInitial Observations:")
print(f"Total number of coins: {len(df)}")
print(f"Years covered: {df['year'].unique()}")
print(f"Most frequent artist: {df['artist'].mode()[0]} (appears {df['artist'].value_counts().max()} times)")
print(f"Highest mintage: {df['mintage'].max()} (Northern Lights, 2004)")
print(f"Lowest mintage: {df['mintage'].min()} (Hopewell Rocks, 2004)")
print(f"Price range: ${df['issue price'].min()} - ${df['issue price'].max()}")

# Final answer not required as the question asks for a description, not a numerical answer.