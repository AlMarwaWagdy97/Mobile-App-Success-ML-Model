import numpy as np
from sklearn.preprocessing import LabelEncoder
def Feature_Encoder(X , cols):
    for c in cols:
        encoder = LabelEncoder()
        X[c] = encoder.fit_transform(X[c])
    return X