import struct
import os
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC

X, y = datasets.make_classification(n_samples=30, n_features=200, random_state=42)
clf = SVC(kernel='rbf', gamma=0.1)
clf.fit(X, y)

gamma = np.float32(clf._gamma)
intercept = np.float32(clf.intercept_[0])

dual_coef = clf.dual_coef_[0].astype(np.float32)
support_vectors = clf.support_vectors_.astype(np.float32)

num_sv = len(dual_coef)
num_features = support_vectors.shape[1]

os.makedirs('training_data', exist_ok=True)
os.chdir('training_data')

with open('svm_model_rv32imafc.bin', 'wb') as f:
    header_format = 'ffii'
    header_bytes = struct.pack(header_format, gamma, intercept, num_sv, num_features)
    f.write(header_bytes)
    
    f.write(dual_coef.tobytes())
    f.write(support_vectors.tobytes())

print(f"Model saved. Support vectors: {num_sv}, Features: {num_features}.")
