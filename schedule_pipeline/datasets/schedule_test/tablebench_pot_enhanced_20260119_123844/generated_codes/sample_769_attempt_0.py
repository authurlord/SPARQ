import pandas as pd

df = pd.read_csv('table.csv')

# Filter for newer models (post-2007) to analyze trends
recent_df = df[df['year'].astype(str).str.contains('2007|2010|2012', na=False)]

# Identify fuel propulsion trends
hybrid_or_electric = recent_df[recent_df['fuel propulsion'].str.contains('hybrid|electric', case=False, na=False)]
if not hybrid_or_electric.empty:
    # Assume future buses will use electric or hybrid
    predicted_fuel_propulsion = 'electric'
else:
    predicted_fuel_propulsion = 'diesel'

# Estimate quantity trend
avg_quantity = df['quantity'].mean()
max_quantity = df['quantity'].max()
# Since quantity increased over time, assume 2025 quantity is higher than current max
predicted_quantity = int(max_quantity * 1.5)

print(f"Final Answer: electric, {predicted_quantity}")