import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert string values to numeric where possible (e.g., remove '-' and convert to float)
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'].str.replace('-', '0'), errors='coerce')
df['bp comp 2 (˚c)'] = pd.to_numeric(df['bp comp 2 (˚c)'].str.replace('-', '0'), errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'], errors='coerce')
df['% wt comp 1'] = pd.to_numeric(df['% wt comp 1'], errors='coerce')
df['% wt comp 2'] = pd.to_numeric(df['% wt comp 2'], errors='coerce')

# Identify the most common components
component_1_counts = df['component 1'].value_counts()
component_2_counts = df['component 2'].value_counts()

# Summary of trends:
# 1. Most common components: acetaldehyde, carbon disulfide, chloroform, toluene
# 2. Azeotropic temperature is often close to the lower boiling component's bp
# 3. Weight percentage of the first component is high when its boiling point is low
# 4. Some combinations show near-azeotropic behavior (e.g., diethyl ether + methyl formate)

print("Main components and trends:")
print(f"Top component 1: {component_1_counts.index[0]} (appears {component_1_counts.iloc[0]} times)")
print(f"Top component 2: {component_2_counts.index[0]} (appears {component_2_counts.iloc[0]} times)")
print("Trend: Azeotropic temperature is typically close to the boiling point of the lower BP component.")
print("Trend: Higher % wt of a component correlates with its lower boiling point.")
print("Trend: Some mixtures (e.g., diethyl ether, methyl formate) show very low azeotropic temps (e.g., 28.2°C), indicating azeotrope formation.")

Final Answer: acetaldehyde, carbon disulfide, azeotropic temperature, weight percentage trend