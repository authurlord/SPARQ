import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Gap' column: extract numeric values from strings like '+3.13' or '–'
def parse_gap(gap):
    if pd.isna(gap) or gap == '–':
        return 0.0
    # Remove the '+' and any extra spaces
    gap_clean = gap.strip().replace('+', '').strip()
    # Convert to float
    try:
        return float(gap_clean)
    except:
        return 0.0

# Apply the parsing function
df['Gap'] = df['Gap'].apply(parse_gap)

# Ensure 'Pos' is numeric for plotting
df['Pos'] = pd.to_numeric(df['Pos'], errors='coerce')

# Drop rows where Pos or Gap is NaN
df = df.dropna(subset=['Pos', 'Gap'])

# Create the line chart
plt.figure(figsize=(12, 6))
plt.plot(df['Pos'], df['Gap'], marker='o', linestyle='-', color='b')
plt.title('Time Difference Between Competitors and First-Place Finisher')
plt.xlabel('Position')
plt.ylabel('Time Difference (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: We are not returning a value, but the chart is generated.
# Since the question asks to draw the chart, no final answer name is needed.
# However, if required to output something, we can just print a confirmation.
print("Final Answer: chart_generated")