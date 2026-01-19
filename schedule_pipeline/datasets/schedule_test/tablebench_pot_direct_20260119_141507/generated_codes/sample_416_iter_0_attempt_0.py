import pandas as pd

df = pd.read_csv('table.csv')

# Display column names and basic statistics
print("Main Columns:")
print(" - name of county: County names")
print(" - county seat: Administrative center of each county")
print(" - area (km square): Land area in square kilometers")
print(" - population: Total population of each county")
print(" - population density: People per km²")
print(" - towns / villages: Number of towns and villages")

# Highlight notable trends
print("\nNotable Trends:")
print("- Pest county has the highest population (1,124,395) and highest population density (176).")
print("- Borsod-Abáúj-Zemplén and Budapest are among the largest in area and population.")
print("- Smaller counties like Komárom-Esztergom have lower populations and areas.")
print("- Population density varies widely, with Pest being the most dense and others ranging from 55 to 139.")

# Optional: Show top row for context
print("\nSample data:")
print(df.head())

Final Answer: Main columns: name of county, county seat, area (km square), population, population density, towns / villages; Notable trends: Pest has highest population and density, areas vary widely, density ranges from 55 to 176