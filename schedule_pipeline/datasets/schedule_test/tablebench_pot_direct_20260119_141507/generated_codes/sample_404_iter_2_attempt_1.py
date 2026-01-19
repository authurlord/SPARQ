import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Provide a detailed overview
print("Column Descriptions:")
print("- Engine type: Type of propulsion system (e.g., rocket, turbojet, turbofan).")
print("- Scenario: Flight condition (e.g., vacuum, Mach 1, cruise).")
print("- SFC in lb/(lbf h): Fuel consumption rate in imperial units.")
print("- SFC in g/(kn s): Fuel consumption rate in metric units.")
print("- Specific impulse (s): Measure of propulsive efficiency; higher values indicate better performance.")
print("- Effective exhaust velocity (m/s): Speed of exhaust gases; correlates with efficiency.")

print("\nNotable Insights:")
print("- Rocket engines (e.g., SSME, NK-33) excel in vacuum environments with high specific impulse (>450) and exhaust velocity (>4400 m/s).")
print("- Turbojets (e.g., J-58, Olympus 593) perform efficiently at high speeds (Mach 2–3.2), with very high exhaust velocities (>29000 m/s).")
print("- Turbofans (e.g., CF6-80C2B1F) are optimized for subsonic cruise with low fuel consumption and high specific impulse (>5900).")
print("- As flight speed increases, specific fuel consumption decreases, indicating improved aerodynamic efficiency at supersonic speeds.")