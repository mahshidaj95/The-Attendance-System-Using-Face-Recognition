import cv2
import face_recognition as fr
import numpy as np


class FaceRecognizer:
    def __init__(self, images):
        self.images = images
        self.faces = self.EncodedFaces()

    def EncodedFaces(self):
        encoded = {}
        for f in self.images:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file(f)
                encoding = fr.face_encodings(face)[0]
                f = f.split('/')[-1]
                encoded[f.split(".")[0]] = encoding
        return encoded

    def ClassifyFace_image(self, im):
        faces_encoded = list(self.faces.values())
        known_face_names = list(self.faces.keys())

        if im:
            img = cv2.imread(im, 1)

            face_locations = fr.face_locations(img)
            unknown_face_encodings = fr.face_encodings(img, face_locations)

            face_names = []
            for face_encoding in unknown_face_encodings:
                matches = fr.compare_faces(faces_encoded, face_encoding)
                name = "Unknown"

                face_distances = fr.face_distance(faces_encoded, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    name = name.split('_')[0]
                face_names.append(name)

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

                    cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)
                while True:
                    cv2.imshow('Result!', img)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break

            return  name

    def ClassifyFace_webcam(self):
        faces_encoded = list(self.faces.values())
        known_face_names = list(self.faces.keys())
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            process_this_frame = True

            if process_this_frame:

                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:

                    matches = fr.compare_faces(faces_encoded, face_encoding)
                    name = "Unknown"

                    face_distances = fr.face_distance(faces_encoded, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        name = name.split('_')[0]
                    face_names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        return name
