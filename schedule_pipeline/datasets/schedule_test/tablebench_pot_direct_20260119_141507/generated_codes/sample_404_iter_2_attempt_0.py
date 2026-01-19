import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display a detailed overview of the table
print("Column Descriptions:")
print("- 'engine type': Type of engine (e.g., rocket, turbojet, turbofan).")
print("- 'scenario': Operating condition (e.g., vacuum, Mach 1, cruise).")
print("- 'sfc in lb/(lbf h)': Specific fuel consumption in imperial units (lower is better).")
print("- 'sfc in g/(kn s)': Specific fuel consumption in metric units (lower is better).")
print("- 'specific impulse (s)': Efficiency measure in seconds (higher is better).")
print("- 'effective exhaust velocity (m/s)': Exhaust speed (higher is better).")

print("\nNotable Insights:")
print("- Turbofan engines (e.g., CF6-80C2B1F) have the highest specific impulse and exhaust velocity, indicating superior efficiency for commercial aviation.")
print("- Rocket engines (e.g., SSME, NK-33) perform best in vacuum but have poor fuel efficiency in atmosphere.")
print("- Supersonic engines (e.g., ramjet, Concorde) achieve high efficiency at high speeds, with exceptional exhaust velocities.")
print("- Specific impulse and exhaust velocity are strong indicators of engine efficiency, with higher values being more favorable.")

Final Answer: engine type, scenario, sfc in lb/(lbf h), sfc in g/(kn s), specific impulse (s), effective exhaust velocity (m / s)