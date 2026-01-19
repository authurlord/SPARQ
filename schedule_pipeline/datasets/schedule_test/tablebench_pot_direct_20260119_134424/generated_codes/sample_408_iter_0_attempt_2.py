import pandas as pd

df = pd.read_csv('table.csv')

# Describe the role of each column
print("Column Roles:")
print("- year: The year the coin was issued.")
print("- design: The thematic subject of the coin (e.g., Niagara Falls, Northern Lights).")
print("- issue: The special feature or technique used in the coin's production (e.g., hologram, colorized).")
print("- artist: The designer responsible for the coin's artwork.")
print("- mintage: The number of coins produced.")
print("- issue price: The original selling price of the coin in USD.")

# Initial observations and trends
print("\nInitial Observations:")
print("- The dataset spans from 2003 to 2005.")
print("- The most frequent issue type is 'hologram' (appears 3 times), followed by 'double image hologram' (2 times).")
print("- José Osio is the most prolific artist, designing 4 out of 6 coins.")
print("- Mintage varies significantly: highest at 35,000 (Diamonds, 2005), lowest at 16,918 (Hopewell Rocks, 2004).")
print("- Issue price is consistently $69.95 for 4 coins and $79.95 for 2 coins, indicating two distinct price tiers.")
print("- The 'double image hologram' feature appears only in 2004 and 2005, suggesting a newer technology introduced later.")

print("Final Answer: year, design, issue, artist, mintage, issue price")