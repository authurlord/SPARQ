import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic info and describe trends
print("Column descriptions:")
print("frequency (Hz): Operating frequency of the transmission line.")
print("r (Ω/km): Resistance per kilometer - increases with frequency.")
print("l (mH/km): Inductance per kilometer - decreases slightly with increasing frequency.")
print("g (μS/km): Conductance per kilometer - increases significantly with frequency, indicating higher dielectric losses.")
print("c (nF/km): Capacitance per kilometer - constant across all frequencies, suggesting frequency independence.")

# Highlight notable trend: resistance and conductance vary with frequency, while capacitance is constant
print("\nNotable trends:")
print("- Resistance (r) increases with frequency.")
print("- Inductance (l) decreases slightly as frequency increases.")
print("- Conductance (g) rises sharply, especially above 100k Hz, indicating increased losses.")
print("- Capacitance (c) remains constant at 51.57 nF/km across all frequencies.")

# Final answer summarizing the main characteristics
Final Answer: resistance increases with frequency, inductance decreases, conductance increases, capacitance is constant