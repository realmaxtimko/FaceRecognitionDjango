from django.http import HttpResponse
from django.shortcuts import render
import face_recognition
import cv2
import numpy as np

def compare_faces_recursive(photo1, photo2, depth=0):

    imag1 = cv2.imdecode(np.frombuffer(photo1, np.uint8), -1)
    imag2 = cv2.imdecode(np.frombuffer(photo2, np.uint8), -1)

    face_locations1 = face_recognition.face_locations(imag1)
    face_encodings1 = face_recognition.face_encodings(imag1, face_locations1)

    face_locations2 = face_recognition.face_locations(imag2)
    face_encodings2 = face_recognition.face_encodings(imag2, face_locations2)

    for (top1, right1, bottom1, left1), face_encoding1 in zip(face_locations1, face_encodings1):
        for (top2, right2, bottom2, left2), face_encoding2 in zip(face_locations2, face_encodings2):
            
            matches = face_recognition.compare_faces(face_encoding1, [face_encoding2])

            if matches[0] == True:
                cv2.rectangle(imag2, (left2, top2), (right2, bottom2), (0, 0, 255), 2)
                return cv2.imencode('.jpg', imag2)[1].tobytes()
            else:
                resized_photo2 = cv2.resize(imag2, (0, 0), fx=0.75, fy=0.75)

    if depth < 5:
        return compare_faces_recursive(photo1, cv2.imencode('.jpg', resized_photo2)[1].tobytes(), depth + 1)

    return None

def index(request):
    if request.method == 'POST':

        photo1 = request.FILES['photo1'].read()
        photo2 = request.FILES['photo2'].read()

        result_image = compare_faces_recursive(photo1, photo2)

        if result_image is not None:
            
            return HttpResponse(result_image, content_type="image/jpg")

    return render(request, 'main/index.html')

def about(request):
    return render(request, 'main/about.html')
