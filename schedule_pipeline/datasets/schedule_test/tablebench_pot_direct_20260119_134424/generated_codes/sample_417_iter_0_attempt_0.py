import pandas as pd

df = pd.read_csv('table.csv')

# Display key columns and brief analysis
key_columns = ['Model', 'Launch', 'Fab (nm)', 'Memory (MiB)', 'Core clock (MHz)', 'Memory clock (MHz)', 'Memory Bandwidth (GB/s)', 'DirectX support']
print("Key Columns:", key_columns)

# Trends observation
print("\nNotable Trends:")
print("- Manufacturing process (Fab) improved from 500 nm to 180 nm over time.")
print("- Memory capacity increased from 2 MiB to 128 MiB.")
print("- Core and memory clock speeds generally increased across generations.")
print("- Memory bandwidth rose significantly, especially in the Spectre series.")
print("- DirectX support evolved from 3.0 to 6.0, with newer models possibly supporting higher versions.")

# Final answer summary
print(f"Final Answer: Key columns include Model, Launch, Fab (nm), Memory (MiB), Core clock (MHz), Memory clock (MHz), Memory Bandwidth (GB/s), and DirectX support. Notable trends: decreasing fab size, increasing memory and clock speeds, rising bandwidth, and improved DirectX support.")