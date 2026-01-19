import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic information
print("Table Characteristics:")
print(f"Number of rows (townships): {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")
print("\nSample data:")
print(df.head())

# Summary statistics for key numerical columns
print("\nSummary statistics for key variables:")
print(df[['pop (2010)', 'land ( sqmi )', 'water (sqmi)']].describe())

# Additional insight: Identify townships with high land area or population
high_land_area = df[df['land ( sqmi )'] > 50]
high_population = df[df['pop (2010)'] > 100]

print(f"\nTownships with land area > 50 sqmi: {len(high_land_area)}")
print(f"Townships with population > 100: {len(high_population)}")
print(f"Top township by land area: {high_land_area.iloc[0]['township']} ({high_land_area.iloc[0]['land ( sqmi )']} sqmi)")
print(f"Top township by population: {high_population.iloc[0]['township']} ({high_population.iloc[0]['pop (2010)']} people)")