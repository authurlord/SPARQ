import pandas as pd

df = pd.read_csv('table.csv')

# Display a detailed overview of the table
print("Column Descriptions:")
print("- 'engine type': Type of propulsion system (e.g., rocket, turbojet, turbofan).")
print("- 'scenario': Operating condition (e.g., vacuum, Mach 1, cruise).")
print("- 'sfc in lb/(lbf h)': Specific fuel consumption in imperial units (lower is better).")
print("- 'sfc in g/(kn s)': Specific fuel consumption in metric units (lower is better).")
print("- 'specific impulse (s)': Measure of efficiency; higher values indicate better fuel efficiency.")
print("- 'effective exhaust velocity (m/s)': Velocity of exhaust gases; higher values mean more efficient propulsion.")

print("\nNotable Insights:")
print("- The Rolls-Royce/SNECMA Olympus 593 engine (Concorde) has the highest specific impulse and exhaust velocity, indicating peak efficiency at Mach 2.")
print("- The J-58 turbojet (SR-71) achieves extremely high exhaust velocity (18,587 m/s), optimized for supersonic flight.")
print("- Rocket engines (e.g., NK-33, SSME) have high specific impulse in vacuum, ideal for space missions.")
print("- Turbofan engines (e.g., CF6-80C2B1F) show low SFC values, making them efficient for commercial airliners at subsonic cruise speeds.")

Final Answer: engine type, scenario, sfc in lb/(lbf h), sfc in g/(kn s), specific impulse (s), effective exhaust velocity (m / s)