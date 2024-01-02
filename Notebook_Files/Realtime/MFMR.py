
#Necessary Libraries
import mediapipe as mp
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Image Merger
#function to merge two frames side by side(i.e. input and output frames )

def addFrame(img1,img2):
    r1,c1 = img1.shape[0:2]
    r2,c2 = img2.shape[0:2]

    result = np.zeros((r2,c1+c2,3),dtype=np.uint8)

    # print(result.shape)

    result[:,:c1,:] = img1
    result[:,c1:c1+c2,:] = img2

    return result

#Video Input - WebCam
#video file
video = cv2.VideoCapture(0)

#mediapipe objects
facemesh = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

landmark1 = []
landmarks = []
face = facemesh.FaceMesh(
    static_image_mode=True,
    min_tracking_confidence=0.6,
    min_detection_confidence=0.6,
    refine_landmarks=True
)

#process each frame
while True:
    ret, frame = video.read()

    if not ret:
        print('Video processing complete.')
        break  

    height, width, channels = frame.shape

    #mediapipe requires RGB format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    #landmarks acquired in op
    op = face.process(rgb)
    
    #if no face detected
    if not op.multi_face_landmarks:
        continue

    #extracting face landmarks
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            if(i.landmark[0] is not None and i.landmark[1] is not None):

                landmarks = []
                landmarks.append(i.landmark[0].y * 480)
                landmarks.append(i.landmark[1].x * 640)
                landmarks.append(i.landmark[2].z)
                landmark1.append(landmarks)

                #Landmarks in the original frame
                # draw.draw_landmarks(
                #     image=frame,
                #     landmark_list=i,
                #     connections=facemesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                # )
                # draw.draw_landmarks(
                #     image=frame,
                #     landmark_list=i,
                #     connections=facemesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                # )
                # draw.draw_landmarks(
                #     image=frame,
                #     landmark_list=i,
                #     connections=facemesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                # )

                #cv2.imshow("Original Video", frame)

                #frame_filename = f"{output_folder}/frame_{int(video.get(cv2.CAP_PROP_POS_FRAMES))}.png"


                #separate window for the face mesh
                mesh_window = np.zeros_like(frame)
                draw.draw_landmarks(
                    image=mesh_window,
                    landmark_list=op.multi_face_landmarks[0],
                    connections=facemesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                draw.draw_landmarks(
                    image=mesh_window,
                    landmark_list=op.multi_face_landmarks[0],
                    connections=facemesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                draw.draw_landmarks(
                    image=mesh_window,
                    landmark_list=op.multi_face_landmarks[0],
                    connections=facemesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                #cv2.imshow("Face Mesh", mesh_window)

                windowimage = addFrame(mesh_window,frame)

                cv2.imshow('MediaPipe Face Mesh', cv2.flip(windowimage, 1))


    # Esc to break loop
    if cv2.waitKey(1) == 27:
        video.release()
        cv2.destroyAllWindows()
        break