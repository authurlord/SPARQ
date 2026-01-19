import pandas as pd

df = pd.read_csv('table.csv')

# Display basic info about the table
print("Table Features:")
print("Columns:")
print("- year: Year of issue")
print("- design: Thematic subject of the coin")
print("- issue: Special feature (e.g., hologram, colorized)")
print("- artist: Designer of the coin")
print("- mintage: Number of coins minted")
print("- issue price: Price at issuance")

print("\nInitial Observations:")
print("- Mintage ranges from 16,918 to 35,000.")
print("- Issue price is $69.95 for simpler designs and $79.95 for advanced features like holograms.")
print("- 2004 had the most coins issued (3).")
print("- José Osio designed 4 out of 6 coins, showing a dominant role.")
print("- Coins with 'double image hologram' or 'hologram' features have higher issue prices.")

# Final answer based on the question
print("Final Answer: year, design, issue, artist, mintage, issue price")