
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
import smtplib

video_cap = cv2.VideoCapture(0)

path = 'Dataset'
Images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

students = classNames.copy()


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(Images)
print('Encoding Complete')

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

lnwriter.writerow(['Name', 'Time'])

while True:
    success, frame = video_cap.read()
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    frameS = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frameS)
    face_encodings = face_recognition.face_encodings(frameS, face_locations)
    face_names = []

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        name = ""
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex]
        else:
            name = 'Unknown'
        face_names.append(name)
        if name in classNames:
            if name in students:
                students.remove(name)
                # print(students)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
f.close()

