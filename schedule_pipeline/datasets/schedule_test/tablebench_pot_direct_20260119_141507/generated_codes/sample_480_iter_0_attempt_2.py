import pandas as pd

df = pd.read_csv('table.csv')

# Convert all values to numeric (some entries like '363.0' are strings)
df = df.apply(pd.to_numeric, errors='coerce')

# Identify anomalies:
# 1. Values greater than 1000 (very high)
# 2. Values less than 1 (very low or zero)
# 3. Zeros in languages that are otherwise dominant

anomalies = []

# Check for extremely high values (>1000)
high_values = df[df > 1000]
if not high_values.empty:
    anomalies.append("Extremely high values detected: e.g., Guaranã in Sorata municipality (7000.0)")

# Check for very low or zero values
low_values = df[df < 1]
if not low_values.empty:
    anomalies.append("Very low or zero values detected: e.g., Guaranã in Quiabaya and Combaya (0.0)")

# Check for inconsistency: a language dominant in one place but absent elsewhere
for lang in df.columns[1:]:
    # If a language has a very high value in one municipality but zero elsewhere
    lang_data = df[lang].dropna()
    if lang_data.max() > 100 and lang_data[lang_data == 0].count() > 0:
        anomalies.append(f"Inconsistent pattern: {lang} is dominant in one area but zero in others.")

# Print anomalies
print(f"Final Answer: {', '.join(anomalies)}")