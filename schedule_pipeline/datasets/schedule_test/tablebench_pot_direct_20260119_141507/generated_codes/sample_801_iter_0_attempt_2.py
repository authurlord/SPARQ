import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'diameter (km)' column and convert to float
diameters = df['diameter (km)'].astype(float)
median_diameter = diameters.median()
print(f"Final Answer: {median_diameter:.1f}")