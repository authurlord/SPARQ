import pandas as pd

df = pd.read_csv('table.csv')

# Display a detailed overview
print("Column Descriptions:")
print(" - 'engine type': Type of engine (e.g., rocket, turbojet, turbofan).")
print(" - 'scenario': Operating condition (e.g., vacuum, Mach 1, cruise).")
print(" - 'sfc in lb/(lbf h)': Fuel consumption rate (lower = more efficient).")
print(" - 'sfc in g/(kn s)': Fuel consumption rate in metric units.")
print(" - 'specific impulse (s)': Efficiency measure (higher = better).")
print(" - 'effective exhaust velocity (m/s)': Speed of exhaust gases (directly related to specific impulse).")

print("\nNotable Insights:")
print(" - Rocket engines (e.g., SSME, NK-33) perform best in vacuum with high specific impulse (>4400 s).")
print(" - The Rolls-Royce/Olympus 593 engine has the highest specific impulse (3012 s) and exhaust velocity (29553 m/s) at Mach 2 cruise.")
print(" - Turbofans (e.g., CF6-80C2B1F) are efficient at subsonic cruise with low fuel consumption (0.605 lb/(lbf h)).")
print(" - The J-58 turbojet achieves high exhaust velocity (18587 m/s) at Mach 3.2, suitable for high-speed flight despite high fuel consumption.")
print(" - Specific impulse and exhaust velocity increase with speed, indicating better performance at supersonic speeds.")

Final Answer: engine type, scenario, sfc in lb / (lbf h), sfc in g / (kn s), specific impulse (s), effective exhaust velocity (m / s)