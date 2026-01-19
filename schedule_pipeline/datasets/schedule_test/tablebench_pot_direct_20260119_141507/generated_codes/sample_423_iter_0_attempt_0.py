import pandas as pd

df = pd.read_csv('table.csv')

# Display a brief description of the table and initial insights
print("Main Columns:")
print(" - city / municipality: Names of the cities or municipalities.")
print(" - no of barangays: Number of administrative divisions (barangays).")
print(" - area (km square): Geographic size in square kilometers.")
print(" - population (2010 census): Population recorded in 2010.")
print(" - pop density (per km square): Population per square kilometer.")
print("\nInitial Insights:")
print(" - Taytay has the highest population density (7447.32 per km²).")
print(" - Antipolo has the largest area (306.1 km²) and moderate density (2214.12 per km²).")
print(" - Smaller areas generally have higher population densities.")
print(" - Binangonan has the most barangays (40), indicating a larger administrative structure.")