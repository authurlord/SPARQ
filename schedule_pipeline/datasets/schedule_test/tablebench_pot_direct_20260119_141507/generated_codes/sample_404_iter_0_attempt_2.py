import pandas as pd

df = pd.read_csv('table.csv')

# Display a detailed overview of the table
print("Column Descriptions:")
print("- 'engine type': Type of engine (e.g., rocket, turbojet, turbofan).")
print("- 'scenario': Operational condition (e.g., vacuum, Mach 1, cruise).")
print("- 'sfc in lb/(lbf h)': Specific fuel consumption in imperial units.")
print("- 'sfc in g/(kn s)': Specific fuel consumption in metric units.")
print("- 'specific impulse (s)': Efficiency measure; higher values indicate better performance.")
print("- 'effective exhaust velocity (m/s)': Speed of exhaust gases; higher values mean better efficiency.")

print("\nNotable Insights:")
print("- The Rolls-Royce/SNECMA Olympus 593 engine has the highest specific impulse (3012) and exhaust velocity (29553 m/s) for Mach 2 cruise, indicating excellent efficiency at supersonic speeds.")
print("- The J-58 turbojet achieves the highest exhaust velocity (18587 m/s) at Mach 3.2, ideal for high-speed aircraft like the SR-71.")
print("- The CF6-80C2B1F turbofan has the highest specific impulse (5950) and exhaust velocity (58400 m/s), optimized for efficient subsonic cruise in commercial aircraft.")
print("- Rocket engines (e.g., NK-33, SSME) have high specific impulse in vacuum, suitable for space launches, but higher fuel consumption compared to jet engines.")

Final Answer: engine type, scenario, sfc in lb / (lbf h), sfc in g / (kn s), specific impulse (s), effective exhaust velocity (m / s)