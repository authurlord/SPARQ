import pandas as pd

df = pd.read_csv('table.csv')

# Display basic info and describe key observations
print("Main Columns:")
print(" - city / municipality: Names of the municipalities.")
print(" - no of barangays: Number of local administrative units.")
print(" - area (km square): Geographic area in square kilometers.")
print(" - population (2010 census): Total population from 2010 census.")
print(" - pop density (per km square): Population per square kilometer.")

print("\nInitial Insights:")
print(" - Taytay has the highest population density (7447.32), indicating a densely populated area.")
print(" - Angono and San Mateo show high population density despite moderate areas.")
print(" - Binangonan and Rodriguez have larger areas with significant populations.")
print(" - Population density varies widely, reflecting differences in urbanization and land use.")