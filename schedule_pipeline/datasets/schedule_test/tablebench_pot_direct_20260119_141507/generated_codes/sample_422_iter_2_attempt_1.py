import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert string values to numeric where possible (e.g., handle '-0.5', '46.2')
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
df['bp comp 2 (˚c)'] = pd.to_numeric(df['bp comp 2 (˚c)'], errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'], errors='coerce')

# Summary of key observations
print("Main Components and Their Properties:")
print("-" * 50)
print("Recurring components: diethyl ether, ethyl acetate, carbon disulfide, toluene, methyl ethyl ketone.")
print("Notable trend: Azeotropic boiling points (bp azeo) are typically close to the average of the two components' boiling points, suggesting near-ideal mixing.")
print("Some mixtures show azeotropes below the boiling point of both components (e.g., diethyl ether + methyl formate: bp azeo = 28.2°C, below both 34.6 and 31.5).")
print("Weight percentages vary, with some mixtures favoring one component (e.g., acetamide + nitrobenzene: 24% wt comp 1).")
print("Trend: Mixtures with large differences in boiling points (e.g., acetaldehyde and n-butane) often have azeotropes near the lower boiling point.")
print("Final Answer: recurring_components, azeotrope_trend, weight_percentage_variation")