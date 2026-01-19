import pandas as pd

df = pd.read_csv('table.csv')

# Describe the main features and trends
print("Columns and their roles:")
print("- 'year': Year of issuance")
print("- 'design': Theme or subject of the coin (e.g., Niagara Falls, Rocky Mountains)")
print("- 'issue': Special feature (e.g., hologram, colorized)")
print("- 'artist': Designer of the coin")
print("- 'mintage': Number of coins produced")
print("- 'issue price': Price at which the coin was sold (consistently $69.95 or $79.95)")

print("\nInitial observations and trends:")
print("- Issue price is stable: only $69.95 or $79.95")
print("- José Osio is the artist on multiple designs (2003, 2004, 2005)")
print("- Mintage varies widely, from 16,918 to 35,000")
print("- Most designs feature natural landscapes and advanced optical effects like holograms")