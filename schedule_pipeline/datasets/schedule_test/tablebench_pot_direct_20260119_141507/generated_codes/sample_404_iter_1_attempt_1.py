import pandas as pd

df = pd.read_csv('table.csv')

# Display a summary of the table with column descriptions and key observations
print("Column Descriptions:")
print("- 'engine type': Type of propulsion engine (e.g., rocket, turbojet, turbofan).")
print("- 'scenario': Operating condition (e.g., vacuum, Mach 1, cruise).")
print("- 'sfc in lb/(lbf h)': Specific fuel consumption in imperial units (lower is better).")
print("- 'sfc in g/(kn s)': Specific fuel consumption in metric units (lower is better).")
print("- 'specific impulse (s)': Efficiency of engine in seconds; higher is better.")
print("- 'effective exhaust velocity (m/s)': Speed of exhaust gases; higher values indicate better efficiency.")

print("\nNotable Insights:")
print("- The Rolls-Royce/SNECMA Olympus 593 (Concorde) has the highest specific impulse (3012 s) and exhaust velocity (29553 m/s), making it highly efficient at Mach 2.")
print("- The J-58 turbojet (SR-71 at Mach 3.2) achieves the highest exhaust velocity (18587 m/s), optimized for supersonic flight.")
print("- Rocket engines (e.g., SSME, NK-33) show high specific impulse in vacuum but high SFC, suitable for launch environments.")
print("- Turbofans (e.g., CF6-80C2B1F) offer excellent fuel efficiency in subsonic cruise (Boeing 747), with SFC of 0.605 lb/(lbf·h).")

Final Answer: engine type, scenario, sfc in lb/(lbf h), sfc in g/(kn s), specific impulse (s), effective exhaust velocity (m / s)