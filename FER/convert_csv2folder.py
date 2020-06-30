import os
import pandas as pd
import cv2
import numpy as np

csv_path = '/home/phatngy/Downloads/fer2013.csv'
out_path = './output'
fer_data = pd.read_csv(csv_path, delimiter=',')
emo_dict = {
    0: 'Angry', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}

labels = fer_data.iloc[:, 0]
# print(labels)
for i, row in (fer_data.iterrows()):
    img = np.asarray(row['pixels'].split(' '), dtype=int).reshape(48, 48)
    emo = row['emotion']
    phase = row['Usage']
    phase_path = os.path.join(out_path, phase)
    if not os.path.exists(phase_path):
        os.makedirs(phase_path)
    emo_path = os.path.join(phase_path, emo_dict[emo])
    if not os.path.exists(emo_path):
        os.makedirs(emo_path)
    img_path = os.path.join(emo_path, str(i)+'.jpg')
    cv2.imwrite(img_path, img)

for phase in os.listdir(out_path):
    phase_path = os.path.join(out_path, phase)
    print(f'----------{phase}----------')
    for emo in os.listdir(phase_path):
        print(f'\t\t{emo}: \t: {len(os.listdir(os.path.join(phase_path, emo)))}')
