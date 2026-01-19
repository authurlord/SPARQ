import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert frequency to numeric (handle '1k', '1m' as 1000, 2000000)
def parse_frequency(freq):
    if freq.endswith('k'):
        return float(freq[:-1]) * 1000
    elif freq.endswith('m'):
        return float(freq[:-1]) * 1000000
    else:
        return float(freq)

df['frequency_hz'] = df['frequency (hz)'].apply(parse_frequency)

# Extract resistance values
resistance = df['r (î / km)'].astype(float)

# Calculate correlation
correlation = df['frequency_hz'].corr(resistance)

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(df['frequency_hz'], resistance, marker='o', linestyle='-', color='b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Resistance (Ω/km)')
plt.title('Resistance vs Frequency')
plt.grid(True)
plt.show()

# Print correlation result
print(f"Final Answer: {correlation:.3f}")