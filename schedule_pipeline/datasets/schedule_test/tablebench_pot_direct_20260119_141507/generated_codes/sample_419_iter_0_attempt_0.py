import pandas as pd

df = pd.read_csv('table.csv')

# Provide a detailed description of the table
print("Main Columns:")
print(" - county: Name of the county.")
print(" - starky, starky %: Number of votes and percentage for Starky candidate.")
print(" - hancock, hancock %: Number of votes and percentage for Hancock candidate.")
print(" - mccain, mccain %: Number of votes and percentage for McCain candidate.")
print(" - total: Total number of votes across all candidates.")

print("\nInitial Insights:")
print("1. McCain dominates in most counties, with 70-79% of the votes.")
print("2. Starky has notable presence in counties like Apache and Pima, with up to 40.95% in Apache.")
print("3. Hancock's share is minimal (usually under 5%), suggesting it is a minor candidate.")
print("4. Maricopa County has the highest total votes (1,163,420), indicating it may be a key area.")
print("5. Data is consistent: percentages sum to approximately 100% in each row, confirming normalization.")