import pandas as pd

df = pd.read_csv('table.csv')

# Describe the main features and roles of each column
print("Column Roles:")
print("- 'year': The year the coin was issued.")
print("- 'design': The thematic design of the coin (e.g., Niagara Falls).")
print("- 'issue': The special feature or technology used (e.g., hologram, colorized).")
print("- 'artist': The artist who created the design.")
print("- 'mintage': The number of coins produced.")
print("- 'issue price': The original selling price of the coin.")

print("\nInitial Observations and Trends:")
print("- Data spans 2003 to 2005.")
print("- José Osio is the most frequent artist, contributing to 4 out of 6 designs.")
print("- 'hologram' and 'double image hologram' are common issue types, with the latter linked to higher prices.")
print("- Mintage ranges from 16,918 to 35,000, with higher mintage often tied to lower issue prices.")
print("- Coins with premium features (e.g., double image hologram, selectively gold plated) have higher issue prices ($79.95).")

# Final Answer: Summarize key insights
print("Final Answer: year, design, issue, artist, mintage, issue price")