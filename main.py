import numpy as np
import cv2
import mediapipe as mp
import time
import pyautogui
from math import sqrt

# initialize blinking tracker variables
COUNTER_TO_BLINK = 0
COUNTER_LEFT = 0
TOTAL_BLINKS = 0

FONT = cv2.FONT_HERSHEY_SIMPLEX

# landmarks from mesh_map.jpg (for facial landmarks)
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

'''function to extract coordinates of landmarks in each frame'''
def landmarksDetection(image, results, draw=False):
    image_height, image_width= image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    #if draw :
     #   [cv2.circle(image, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
    return mesh_coordinates

'''Euclidean distance to calculate the distance between the two points'''
def euclaideanDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

'''Extract eye landmarks for both eyes, calculate their height and width, and calculate the ratio
of width to height. Closed eyes will have a larger ratio while open eyes will have a smaller ratio.
The difference in height-width ratio between the eyes determines whether user is winking or not
(larger ratio implies winking, smaller ratio implies blinking or open eyes)'''
def blinkRatio(image, landmarks, right_indices, left_indices):
    # get right and left eye landmarks
    right_eye_landmark1 = landmarks[right_indices[0]]
    right_eye_landmark2 = landmarks[right_indices[8]]

    right_eye_landmark3 = landmarks[right_indices[12]]
    right_eye_landmark4 = landmarks[right_indices[4]]

    left_eye_landmark1 = landmarks[left_indices[0]]
    left_eye_landmark2 = landmarks[left_indices[8]]

    left_eye_landmark3 = landmarks[left_indices[12]]
    left_eye_landmark4 = landmarks[left_indices[4]]

    # calculate eye height and eye width of right eye
    right_eye_horizontal_distance = euclaideanDistance(right_eye_landmark1, right_eye_landmark2)
    right_eye_vertical_distance = euclaideanDistance(right_eye_landmark3, right_eye_landmark4)

    # calculate eye height and eye width of left eye
    left_eye_vertical_distance = euclaideanDistance(left_eye_landmark3, left_eye_landmark4)
    left_eye_horizobtal_distance = euclaideanDistance(left_eye_landmark1, left_eye_landmark2)

    # prevent division by 0 error
    if left_eye_vertical_distance == 0:
        left_eye_vertical_distance = 0.000001
    if right_eye_vertical_distance == 0:
        right_eye_vertical_distance = 0.000001

    # calculate ratio of eye height and eye width for both eyes
    right_eye_ratio = right_eye_horizontal_distance/right_eye_vertical_distance
    left_eye_ratio = left_eye_horizobtal_distance/left_eye_vertical_distance

    # calculate average eye ratio
    eyes_ratio = (right_eye_ratio+left_eye_ratio)/2

    return eyes_ratio

# initialize mediapipe with the face mesh detection module, set detection and tracking confidence
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

# initialize mediapipe with drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)


# initialize video capture through webcam
cap = cv2.VideoCapture(0)

# while camera is running
while cap.isOpened():
    # Read the frame and get the success, which tells if the frame was read correctly
    success, image = cap.read()

    start = time.time()

    # makes the image a 3 channel image
    image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) #flipped for selfie view

    # make image read-only to avoid errors in data processing
    image.flags.writeable = False

    results = face_mesh.process(image)    # generates landmarks

    image.flags.writeable = True

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    img_h , img_w, img_c = image.shape

    face_2d = []
    face_3d = []


    # if landmarks are detected
    if results.multi_face_landmarks:
        # get mesh coordinates from the frame
        mesh_coordinates = landmarksDetection(image, results, True)

        # calculate the ratio of width to height of the eyes
        eyes_ratio = blinkRatio(image, mesh_coordinates, RIGHT_EYE, LEFT_EYE)

        # if eyes are closed in the frame then increment the blink frame counter
        if eyes_ratio > 4:
            COUNTER_TO_BLINK += 1

        else:
            # 8 frames in a row is a long blink and codes for a right click
            if COUNTER_TO_BLINK > 8:
                TOTAL_BLINKS += 1
                pyautogui.rightClick()

            # 4 frames in a row is a medium blink and codes for a left click
            elif COUNTER_TO_BLINK > 4:
                TOTAL_BLINKS += 1
                pyautogui.click()

            # reset the blink frame counter
            COUNTER_TO_BLINK = 0


        # show total number of blinks
        cv2.putText(image, f'Total Blinks: {TOTAL_BLINKS}', (30, 300), FONT, 1, (0, 255, 0), 2)

        # get 3D face coordinates
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    # get the specific mesh coordinates for the nose
                    if idx ==1:
                        nose_2d = (lm.x * img_w,lm.y * img_h)
                        nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                    # get the mesh coordinates for all other landmarks
                    x,y = int(lm.x * img_w),int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))


            #Convert landmarks list to numpy array
            face_2d = np.array(face_2d,dtype=np.float64)
            face_3d = np.array(face_3d,dtype=np.float64)


            # the camera matrix and distortion matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length,0,img_h/2],
                                  [0,focal_length,img_w/2],
                                  [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)

            # get rotation and translation matrix, and if it was successful
            # Perspective-n-Point is the problem of estimating the pose of a calibrated camera given a
            # set of n 3D points in the world and their corresponding 2D projections in the image.
            success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


            # convert rotation vector to rotation matrix, getting angle of face
            rmat,jac = cv2.Rodrigues(rotation_vec)

            # get the yaw, pitch, and roll angles as values between 0 and 1
            angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

            # get the x, y, and z as angles in degrees
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # set the speed of the mouse in x and y axis
            x_speed = 0
            y_speed = 0

            # based on the angle of the face, increase the speed in the respective axis

            if x > 15 or x < -8:
                x_speed = 3

            elif x > 10 or x < -5:
                x_speed = 2

            if y < -15 or y > 12:
                y_speed = 3

            elif y < -10 or y > 10:
                y_speed = 2

            pyautogui.move(y_speed*y,-1*x_speed*x)

            # show the text on the screen indicating the direction of the face
            if y < -10:
                text="Looking Left"
            elif y > 10:
                text="Looking Right"
            elif x < -10:
                text="Looking Down"
            elif x > 10:
                text="Looking Up"
            else:
                text="Forward"

            # create projection from the nose surface to the eyes
            nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

            # draw the normal line from the nose surface
            p1 = (int(nose_2d[0]),int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))
            cv2.line(image,p1,p2,(255,0,0),3)

            cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # record time since the frame was first shown to the screen
        end = time.time()
        totalTime = end-start

        # show FPS
        fps = 1/totalTime
        #print("FPS: ",fps)

        cv2.putText(image,f'FPS: {int(fps)}',(20,450),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

    # display the resulting frame
    cv2.imshow('Head Pose Detection',image)
    if cv2.waitKey(5) & 0xFF ==27:
        break

# After the loop release the cap object (free up camera resources)
cap.release()