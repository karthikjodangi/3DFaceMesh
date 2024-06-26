import mediapipe as mp
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import shutil
#mediapipe object

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#for c=webcam input

cap = cv2.VideoCapture(0)

# input video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = 60

#processing each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  

    #mediapipe requires RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #landmarks acquired in results
    results = face_mesh.process(frame_rgb)

    #if no face detected
    if not results.multi_face_landmarks:
        continue

    #extracting landmarks
    landmarks_3d = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y, z = landmark.x, landmark.y, landmark.z
                landmarks_3d.append((x, y, z))

    landmarks_3d = np.array(landmarks_3d)

    # Delaunay triangulation
    triangulation = Delaunay(landmarks_3d[:, :2])  

    # 3D surface plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # triangulation in 3D
    ax.plot_trisurf(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], triangles=triangulation.simplices,
                    color='lightblue', edgecolor='black')

    ax.set_title('3D triangulated mesh')
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # face orientation
    ax.view_init(elev=-90, azim=-90)  

    # image with facial keypoints
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(frame_rgb)
    ax2.set_title('Face input')
    ax2.axis('off')
    plt.axis('off')
    ax.set_axis_off()

    # frame with the triangulation
    temp_frame_path = 'temp_frame.png'
    plt.savefig(temp_frame_path)
    plt.close()

    # output frame
    temp_frame = cv2.imread(temp_frame_path)

    cv2.imshow('Face Mesh', temp_frame)

    if cv2.waitKey(10) & 0xFF == 27:  
        break

# Release resources
cap.release()
cv2.destroyAllWindows()