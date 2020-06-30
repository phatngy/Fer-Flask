import os
import glob
import shutil

in_path = '/u01/DATA/DATA/face_quality/vinfast/out'
out_path = '/u01/DATA/DATA/face_quality/vinfast/out/full'
def merge_mm(in_path, out_path):
    mm_paths = glob.glob(os.path.join(in_path, '0*'))
    for mm_path in mm_paths:
        for label in os.listdir(mm_path):
            label_path = os.path.join(mm_path, label)
            label_dst = os.path.join(out_path, label)
            if not os.path.isdir(label_dst):
                os.makedirs(label_dst)
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                shutil.copy(img_path, label_dst)
            print(f'{label} = \t', len(os.listdir(label_dst)))






## merge following month            etc: 03, 04, 05



# merge following folder           etc: vinmart, vin bdi
def merge_vin():
    ls_in_path = [
            '/u01/DATA/DATA/face_quality/vinfast/out/split',
            '/u01/DATA/DATA/face_quality/vinmart/split/0604',
            '/u01/DATA/DATA/face_quality/vinbdi/step5_split'
        ]

    out_path = '/u01/DATA/DATA/face_quality/split'

    for in_path in ls_in_path:
        for phase in os.listdir(in_path):
            phase_path = os.path.join(in_path, phase)
            out_phase = os.path.join(out_path, phase)
            if not os.path.isdir(out_phase):
                os.makedirs(out_phase)
            for label in os.listdir(phase_path):
                label_path = os.path.join(phase_path, label)
                out_label = os.path.join(out_phase, label)
                if not os.path.isdir(out_label):
                    os.makedirs(out_label)
                for img in os.listdir(label_path):
                    img_path = os.path.join(label_path, img)
                    shutil.copy(img_path, out_label)