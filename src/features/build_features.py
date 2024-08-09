from src.features.feature_selection import get_features
import pandas as pd
from sklearn.model_selection import train_test_split

X,y = get_features()
print(X.columns)
print(y)