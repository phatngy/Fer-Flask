import cv2
import torch
import numpy as np



img = cv2.imread('./output/Training/Angry/0.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img, [2, 0, 1])
a = img[0]
print(a)
b = img[1]
c = img[2]
x = np.concatenate(([a], [b], [c]), axis=0)
print(x.shape)
# tensor = torch.from_numpy(img)
# tensor = tensor.view
print()