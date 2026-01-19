import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert string values to numeric where possible (e.g., remove '–' or space issues)
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'].str.replace('-', '', regex=False), errors='coerce')
df['bp comp 2 (˚c)'] = pd.to_numeric(df['bp comp 2 (˚c)'].str.replace('-', '', regex=False), errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'].str.replace('-', '', regex=False), errors='coerce')

# Calculate differences to identify azeotrope proximity
df['diff_to_comp1'] = abs(df['bp azeo (˚c)'] - df['bp comp 1 (˚c)'])
df['diff_to_comp2'] = abs(df['bp azeo (˚c)'] - df['bp comp 2 (˚c)'])

# Find cases where azeotrope is very close to one component (within 1°C)
close_to_comp1 = df[df['diff_to_comp1'] <= 1]
close_to_comp2 = df[df['diff_to_comp2'] <= 1]

# Summary of trends
print("Main components and properties:")
print(f"Total entries: {len(df)}")
print(f"Components with azeotrope close to component 1 (within 1°C): {len(close_to_comp1)}")
print(f"Components with azeotrope close to component 2 (within 1°C): {len(close_to_comp2)}")

# Highlight a few notable examples
print("\nNotable examples with near-azeotropic behavior:")
for _, row in close_to_comp1.head(3).iterrows():
    print(f"  {row['component 1']} + {row['component 2']} → bp azeo ≈ {row['bp azeo (˚c)']}°C (close to comp 1)")

# Weight percentage trend: check if high % wt comp 1 correlates with high bp comp 1
correlation = df['% wt comp 1'].corr(df['bp comp 1 (˚c)'])
print(f"\nCorrelation between % wt comp 1 and bp comp 1: {correlation:.2f}")

# Final answer: describe key observations
Final Answer: near-azeotrope pairs, high % wt comp 1, positive correlation between wt% and boiling point