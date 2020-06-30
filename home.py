from flask import Flask, request, render_template
import io
import dlib
import cv2
import numpy as np
from PIL import Image
from FER.predict import Classify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    face_detector = dlib.get_frontal_face_detector()
    model = Classify()
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # print('files uploaded info: \t', request.files)
        if 'file' not in request.files:
            return 'file not uploaded'
        # print(request.files)
        else:
            file = request.files['file']
            img_byte = file.read()
            img = np.asarray(Image.open(io.BytesIO(img_byte)))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray, 1)
            if len(faces) >= 1:
                x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
                
                crop_face = np.zeros((int(x2-x1), int(y2-y1)), dtype=np.uint8)
                crop_face = gray[y1:y2, x1:x2]
                crop_face = np.concatenate(([crop_face], [crop_face], [crop_face]), axis=0)
                # print(crop_face.shape)
                crop_face = np.transpose(crop_face, [1, 2, 0])
                score, pred = model.predict(crop_face)

            # print(score)
                txt = '{} - {:.2f}'.format(pred, score)
                # print(txt)
                t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frm = cv2.putText(img, txt, (x1,y1+t_size[1]+4),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.imwrite('2.jpg', frm)
                return render_template('results.html', emotion=pred)
            else:
                return 'no face to predict'
                # return
if __name__ == "__main__":
    app.run(debug=True)
