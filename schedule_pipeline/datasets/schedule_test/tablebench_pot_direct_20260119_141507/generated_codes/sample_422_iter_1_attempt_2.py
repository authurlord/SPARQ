import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert string values to numeric where possible (e.g., '-0.5' to -0.5)
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
df['bp comp 2 (˚c)'] = pd.to_numeric(df['bp comp 2 (˚c)'], errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'], errors='coerce')

# Calculate average boiling point of components
df['avg_bp'] = (df['bp comp 1 (˚c)'] + df['bp comp 2 (˚c)']) / 2

# Determine if azeotrope is minimum or maximum
df['azeo_type'] = 'min' if df['bp azeo (˚c)'] < df['avg_bp'] else 'max'

# Count occurrences of min and max azeotropes
min_azeo_count = df[df['azeo_type'] == 'min'].shape[0]
max_azeo_count = df[df['azeo_type'] == 'max'].shape[0]

# Highlight dominant components (weight % > 80)
dominant_components = df[(df['% wt comp 1'] > 80) | (df['% wt comp 2'] > 80)]

# Print summary
print("Main components and their properties:")
print("Azeotrope types observed: min (", min_azeo_count, ") and max (", max_azeo_count, ")")
print("Dominant components (weight > 80%):")
print(dominant_components[['component 1', 'component 2', '% wt comp 1', '% wt comp 2']])

print(f"Final Answer: min_azeo_count, max_azeo_count, dominant_components")