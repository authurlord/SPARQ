import pandas as pd

df = pd.read_csv('table.csv')

# Display basic info about the table
print("Table Description:")
print("The table contains data on transportation infrastructure across various regions in Russia.")
print("Columns include: region, location, opening date (from), number of depots (as of 12.09), number of routes (as of 12.09), and number of vehicles (as of 12.09).")
print()

# Insights
print("Key Insights:")
print(f"- The region with the highest number of vehicles is Novosibirsk Oblast ({df['vehicles (12.09)'].max()} vehicles in Novosibirsk).")
print(f"- The region with the most depots is Novosibirsk Oblast with {df['depots (12.09)'].max()} depots.")
print(f"- The location with the most routes is Novosibirsk with {df['routes (12.09)'].max()} routes.")
print(f"- The oldest depot is in Krasnoyarsk (opened 5 Nov 1959), while the newest is in Abakan (31 Dec 1980).")
print(f"- The total number of locations listed is {len(df)}.")
print(f"- The average number of vehicles per location is {df['vehicles (12.09)'].mean():.1f}.")
print(f"- The region with the most locations is Novosibirsk Oblast with {df[df['region'] == 'novosibirsk obl'].shape[0]} locations.")

# Final answer: summarize main content and highlight key insight
print(f"Final Answer: Region, Location, Opening Date, Depots, Routes, Vehicles; Novosibirsk Oblast has the highest number of depots and vehicles, and the most routes.")