import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic information about the table
print("Main Characteristics of the Table:")
print(f"Number of rows (townships): {len(df)}")
print(f"Columns: {', '.join(df.columns)}")
print("\nKey Insights:")
print(f"- Most populous township: Trenton with {df['pop (2010)'].max()} people.")
print(f"- Largest land area: Tri with {df['land ( sqmi )'].max()} sqmi.")
print(f"- Smallest population: Taylor Butte with {df['pop (2010)'].min()} people.")
print(f"- Largest water area: Twin Valley with {df['water (sqmi)'].max()} sqmi.")
print(f"- Geographic coverage spans from {df['latitude'].min()} to {df['latitude'].max()} latitude and {df['longitude'].min()} to {df['longitude'].max()} longitude.")