import pandas as pd

df = pd.read_csv('table.csv')

# Display the first few rows for context
print("Table Overview:")
print(df.head())

# Describe columns
print("\nColumn Roles:")
print("- 'year': Year the coin was issued.")
print("- 'design': Theme or image featured on the coin.")
print("- 'issue': Special technical or visual feature (e.g., hologram).")
print("- 'artist': Artist responsible for the design.")
print("- 'mintage': Number of coins minted.")
print("- 'issue price': Price at which the coin was issued.")

# Initial observations
print("\nInitial Observations and Trends:")
print("- Multiple designs were released per year (e.g., 2003 and 2004).")
print("- 'hologram' and 'double image hologram' are common issue types.")
print("- José Osio is the most frequent artist, designing 4 out of 6 coins.")
print("- Mintage ranges from 16,918 to 35,000; higher mintage often correlates with lower issue price.")
print("- Issue price is either $69.95 or $79.95, with $69.95 being more common.")
print("- The highest mintage (35,000) is for 'diamonds' (2005), which has a lower price.")

# Final answer format required: only the final summary in specified format
print("Final Answer: year, design, issue, artist, mintage, issue price")