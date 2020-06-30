import os
import glob
import numpy as np
from sklearn.svm import SVC
from PIL import Image
import pickle
import joblib
import random

imgs_train = './output/Training'
X = []
y = []
for label in os.listdir(imgs_train):
    count = 0
    label_path = os.path.join(imgs_train, label)
    imgs_path = glob.glob(os.path.join(label_path, "*.jpg"))
    random.shuffle(imgs_path)
    for path in imgs_path:
        if count >= 500:
            continue
        X.append(np.asarray(Image.open(path)).reshape(-1))
        y.append(label)
        count += 1
        
print(len(X))
print(len(y))
print(X[1].shape)
clf = SVC(gamma=1e-3, C=1e2)
clf.fit(X, y)

joblib.dump(clf, 'model.joblib')

