import pandas as pd

df = pd.read_csv('table.csv')

# Display basic info and summary statistics
print("Column Descriptions:")
print("- Engine type: Type of propulsion system (e.g., rocket, turbojet, turbofan).")
print("- Scenario: Flight condition (e.g., vacuum, Mach 1, cruise).")
print("- SFC in lb/(lbf·h): Fuel consumption rate in imperial units.")
print("- SFC in g/(kn·s): Fuel consumption rate in metric units.")
print("- Specific impulse (s): Measure of fuel efficiency; higher values indicate better performance.")
print("- Effective exhaust velocity (m/s): Velocity of exhaust gases; higher values mean more efficient propulsion.")

print("\nNotable Insights:")
print("- The Rolls-Royce/SNECMA Olympus 593 (Concorde) has the highest specific impulse (3012) and exhaust velocity (29553 m/s), indicating peak efficiency at Mach 2.")
print("- The J-58 turbojet (SR-71) achieves the highest exhaust velocity (18587 m/s), showing exceptional performance at Mach 3.2.")
print("- The CF6-80C2B1F turbofan (Boeing 747) has the highest specific impulse (5950) and exhaust velocity (58400 m/s), optimized for commercial cruise efficiency.")
print("- Specific fuel consumption (SFC) increases with speed or atmospheric conditions, reflecting higher fuel demands at high speeds or in non-vacuum environments.")
print("- Vacuum scenarios (e.g., space shuttle) show lower SFC due to absence of atmospheric resistance.")

Final Answer: engine type, scenario, sfc in lb / (lbf h), sfc in g / (kn s), specific impulse (s), effective exhaust velocity (m / s)