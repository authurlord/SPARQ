import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Gap' column: remove '+' and convert to float
def parse_gap(gap):
    if pd.isna(gap) or gap == '–':
        return 0.0
    try:
        # Remove the '+' sign and convert to float
        return float(gap.strip().replace('+', '').strip())
    except:
        return 0.0

# Apply the parsing function
df['Gap'] = df['Gap'].apply(parse_gap)

# Extract position and gap for plotting
positions = df['Pos'].astype(int)
gaps = df['Gap']

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(positions, gaps, marker='o', linestyle='-', color='b')
plt.title('Time Difference (Gap) Between Competitors and First-Place Finisher')
plt.xlabel('Position')
plt.ylabel('Time Difference (seconds)')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()