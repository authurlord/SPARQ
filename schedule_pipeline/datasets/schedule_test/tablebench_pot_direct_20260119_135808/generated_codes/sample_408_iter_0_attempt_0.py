import pandas as pd

df = pd.read_csv('table.csv')

# Describe the role of each column
print("Column Roles:")
print("- year: The year the coin was issued.")
print("- design: The thematic design featured on the coin (e.g., Niagara Falls).")
print("- issue: The special feature or technique used in the coin's production (e.g., hologram).")
print("- artist: The artist responsible for the coin's design.")
print("- mintage: The total number of coins produced.")
print("- issue price: The original selling price of the coin in USD.")

# Initial observations and trends
print("\nInitial Observations and Trends:")
print("- The majority of coins were issued between 2003 and 2005.")
print("- 'hologram' and 'double image hologram' are common issue types, appearing in multiple years.")
print("- José Osio is the most frequent artist, designing 4 out of 6 coins.")
print("- Mintage numbers vary significantly: highest at 35,000 (2005 Diamonds) and lowest at 16,918 (2004 Hopewell Rocks).")
print("- Issue price is consistent at $69.95 for most coins, except for those with 'hologram' or 'selectively gold plated' features, which are priced higher ($79.95).")
print("- The 'Diamonds' (2005) coin has the highest mintage (35,000), while 'Hopewell Rocks' (2004) has the lowest (16,918).")

# Final summary
print("\nFinal Answer: year, design, issue, artist, mintage, issue price")