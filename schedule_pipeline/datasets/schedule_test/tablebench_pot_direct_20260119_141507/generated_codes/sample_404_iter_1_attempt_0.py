import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display a detailed overview of the table
print("Column Descriptions:")
print("- 'engine type': Type of propulsion system (e.g., rocket, turbojet, turbofan).")
print("- 'scenario': Operating condition (e.g., vacuum, Mach 1, cruise).")
print("- 'sfc in lb/(lbf h)': Specific fuel consumption in imperial units (lower is better).")
print("- 'sfc in g/(kn s)': Specific fuel consumption in metric units (lower is better).")
print("- 'specific impulse (s)': Efficiency measure; higher values indicate better fuel efficiency.")
print("- 'effective exhaust velocity (m/s)': Speed of exhaust gases; higher values mean better performance.")

print("\nNotable Insights:")
print("- The Rolls-Royce/Olympus 593 engine (Concorde at Mach 2) has the highest specific impulse and exhaust velocity, indicating peak efficiency at supersonic speeds.")
print("- The J-58 turbojet (SR-71 at Mach 3.2) achieves very high exhaust velocity, enabling high-speed flight.")
print("- Rocket engines (e.g., NK-33, SSME) perform well in vacuum but have lower specific impulse than jet engines in cruise conditions.")
print("- Turbofan engines (e.g., CF6-80C2B1F) are optimized for efficient subsonic flight, with excellent specific impulse and exhaust velocity.")

Final Answer: engine type, scenario, sfc in lb / (lbf h), sfc in g / (kn s), specific impulse (s), effective exhaust velocity (m / s)