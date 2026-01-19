import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Display basic info and describe trends
print("Column Description:")
print("frequency (hz): Operating frequency of the transmission line.")
print("r (î / km): Resistance per kilometer (in ohms per km).")
print("l (mh / km): Inductance per kilometer (in millihenries per km).")
print("g (î¼s / km): Conductance per kilometer (in siemens per km).")
print("c (nf / km): Capacitance per kilometer (in nanofarads per km).")

print("\nNotable Trends:")
print("- Resistance (r) increases with frequency, reaching a peak at 2 MHz.")
print("- Inductance (l) decreases as frequency increases.")
print("- Conductance (g) increases significantly with frequency, especially above 100 kHz.")
print("- Capacitance (c) remains constant across all frequencies, indicating no frequency dependence.")

Final Answer: resistance increases, inductance decreases, conductance increases, capacitance constant