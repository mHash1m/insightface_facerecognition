import os
import json
import time
import cv2
import onnxruntime as ort
import numpy as np
from numpy.linalg import norm
from pathlib import Path
import argparse

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def sim(embedding_1, embedding_2):
    cosine = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
    return float("{:.2f}".format(cosine))

def register_faces(app, target):
    people = os.listdir(target)
    people_embs = dict()
    for person in people:
        person_dir = Path(os.path.join(target, person))
        imgs_list = os.listdir(person_dir)
        embedding = []
        for i in range(len(imgs_list)):
            img = cv2.imread(str(person_dir / imgs_list[i]))
            faces = app.get(img)
            embedding.append(faces[0]["embedding"])
            print(imgs_list[i], len(faces))
            rimg = app.draw_on(img, faces)
            out_dir = Path(f"./results/reg_face_outputs/{person}")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(str(out_dir / imgs_list[i]), rimg)
        people_embs[person] = np.average(embedding, axis=0).tolist()
    json.dump(people_embs, open("registered_faces.json", 'w'))

def run_for_video(app, target, register):
    start_time = time.time()
    cap = cv2.VideoCapture(target)
    fps, width, height = (int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = Path(target).stem
    out = cv2.VideoWriter(f"./results/{file_name}_output.mp4", fourcc, fps/5, (width, height))
    counter = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
                break
        faces = app.get(frame)
        for face in faces:
            face_emb = face["embedding"]
            face_age = face["age"]
            gender = "M" if face["gender"] == 1 else "F"
            highest_sim = ("mike", -1)
            for key, val in register.items():
                cos_sim = sim(face_emb, val)
                if  cos_sim > highest_sim[1]:
                    highest_sim = [key, cos_sim]
            x1, y1, x2, y2 = s = [int(px) for px in face["bbox"]]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(127,69,18),3)
            cv2.putText(frame,f"{gender}:{face_age}",(x1,y2+24),5,0.7,(255,178,102))
            # print("Averaged =", highest_sim)
            if highest_sim[1] >= 0.3:
                cv2.putText(frame,f"{highest_sim[0]}[{highest_sim[1]}]",(x1,y2+12),5,0.7,(255,178,102))
        out.write(frame)
        print(f"Frame # {counter} predicted!")
        counter+=1
    print("Video Width = ", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), ", Video Height = ", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Average FPS: ", counter / (time.time() - start_time))
    cap.release()
    cv2.destroyAllWindows()

def run_for_webcam(app, target, register, show):
    # define a video capture object
    vid = cv2.VideoCapture(int(target))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = Path(target).stem
    out = cv2.VideoWriter(f"./results/webcam_output.mp4", fourcc, 5, (640,480))
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        faces = app.get(frame)
        for face in faces:
            face_emb = face["embedding"]
            face_age = face["age"]
            gender = "M" if face["gender"] == 1 else "F"
            highest_sim = ("mike", -1)
            for key, val in register.items():
                cos_sim = sim(face_emb, val)
                if  cos_sim > highest_sim[1]:
                    highest_sim = [key, cos_sim]
            x1, y1, x2, y2 = s = [int(px) for px in face["bbox"]]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(127,69,18),3)
            cv2.putText(frame,f"{gender}:{face_age}",(x1,y2+24),5,0.7,(255,178,102))
            # print("Averaged =", highest_sim)
            if highest_sim[1] >= 0.3:
                cv2.putText(frame,f"{highest_sim[0]}[{highest_sim[1]}]",(x1,y2+12),5,0.7,(255,178,102))
        # Display the resulting frame
        if show:
            cv2.imshow('frame', frame)
        out.write(frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def run_for_image(app, target, register):
    file_name = Path(target).stem
    img = cv2.imread(target)
    faces = app.get(img)
    for face in faces:
        face_emb = face["embedding"]
        face_age = face["age"]
        gender = "M" if face["gender"] == 1 else "F"
        highest_sim = ("mike", -1)
        for key, val in register.items():
            cos_sim = sim(face_emb, val)
            if  cos_sim > highest_sim[1]:
                highest_sim = [key, cos_sim]
        x1, y1, x2, y2 = s = [int(px) for px in face["bbox"]]
        cv2.rectangle(img,(x1,y1),(x2,y2),(127,69,18),3)
        cv2.putText(img,f"{gender}:{face_age}",(x1,y2+24),5,0.7,(255,178,102))
        print("Averaged =", highest_sim)
        if highest_sim[1] >= 0.3:
            cv2.putText(img,f"{highest_sim[0]}[{highest_sim[1]}]",(x1,y2+12),5,0.7,(255,178,102))
    cv2.imwrite(f"./results/output_{file_name}.jpg", img)

parser=argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, required=True, help="'register', 'inference image' or 'inference video'")
parser.add_argument("-tg", "--target", type=str, required=True, help="path to vid, image or dir")
parser.add_argument("-s", "--show", default=False, action='store_true', help="show cam feed inference")
args=parser.parse_args()
task = args.task
target = args.target
show = args.show

app = FaceAnalysis(providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

try:
    os.mkdir("./results")
except OSError as error:
    print(str(error)[10:])

if task == "register":
    register_faces(app, target)
elif task == "inference image":
    register = json.load(open("./registered_faces.json"))
    run_for_image(app, target, register)
elif task == "inference video":
    register = json.load(open("./registered_faces.json"))
    run_for_video(app, target, register)
elif task == "inference cam":
    register = json.load(open("./registered_faces.json"))
    run_for_webcam(app, target, register, show)
else:
    print("UNKNOWN TASK, should be 'register', 'inference image' or 'inference video'")