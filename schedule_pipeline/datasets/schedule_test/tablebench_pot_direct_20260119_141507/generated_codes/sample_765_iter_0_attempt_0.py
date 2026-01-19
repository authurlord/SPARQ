But the Year_2 column is not directly aligned with the Year column. For example, the row with Year=1820 has Year_2=249,187 (for 1950). So it seems that the data is structured as:
- Extract all rows where the year is >= 1950
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df =