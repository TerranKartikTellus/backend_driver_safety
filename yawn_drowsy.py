import asyncio
import json
import websockets
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import json
from datetime import datetime
import websocket
import json
import requests
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

detectorgffd = dlib.get_frontal_face_detector()


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector,
                             translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector,
                             translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]

    return (x, y)

def sound_alarm(path):
 now = datetime.now()
 current_time = now.strftime("%H:%M:%S")
 print("ALERT: Current Time =", current_time)


def get_landmarks(detector, im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def mouth_open(detector, image):
    landmarks = get_landmarks(detector, image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


def sound_alarm(path):
  pass


def eye_aspect_ratio(eye):
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])
  C = dist.euclidean(eye[0], eye[3])
  ear = (A + B) / (2.0 * C)
  return ear


def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(
            39).y, landmarks.part(37).x, landmarks.part(37).y, landmarks.part(38).x, landmarks.part(38).y
        right_eye = landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(
            45).y, landmarks.part(43).x, landmarks.part(43).y, landmarks.part(44).x, landmarks.part(44).y
        left_eye_aspect_ratio = (
            left_eye[5] - left_eye[1] + left_eye[7] - left_eye[3]) / (2 * (left_eye[2] - left_eye[0]))
        right_eye_aspect_ratio = (
            right_eye[5] - right_eye[1] + right_eye[7] - right_eye[3]) / (2 * (right_eye[2] - right_eye[0]))
        eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
        if eye_aspect_ratio < 0.25:
            return True
    return False


async def send_drowsiness_status(websocket, path):
  # detector = dlib.get_frontal_face_detector()
  yawns = 0
  yawn_status = False
  ap = argparse.ArgumentParser()
  ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                  help="path to facial landmark predictor")
  ap.add_argument("-a", "--alarm", type=str, default="alarm.wav",
                  help="path alarm .WAV file")
  ap.add_argument("-w", "--webcam", type=int, default=0,
                  help="index of webcam on system")
  args = vars(ap.parse_args())
  EYE_AR_THRESH = 0.30
  EYE_AR_CONSEC_FRAMES = 25
  COUNTER = 0
  ALARM_ON = False

  print("[INFO] loading facial landmark predictor...")
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(args["shape_predictor"])
  (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
  (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
  print("[INFO] starting video stream thread...")
  # vs = VideoStream(src=args["webcam"]).start()
  vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  time.sleep(1.0)

  while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    data = {'drowsy': 0, 'yawn': 0, 'time': current_time}
    ret, frame = vs.read()
    if not ret:
        break
      
    frame = imutils.resize(frame, width=550)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    image_landmarks, lip_distance = mouth_open(detector, frame)
    prev_yawn_status = yawn_status
    if lip_distance > 37:
        yawn_status = True
        sound_alarm('/')
        cv2.putText(frame, "Subject is Yawning", (50, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        output_text = " Yawn Count: " + str(yawns + 1)
        data['yawn'] = 1
        cv2.putText(frame, output_text, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

    else:
        data['yawn'] = 0
        yawn_status = False

    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    for rect in rects:
      shape = predictor(gray, rect)
      shape = face_utils.shape_to_np(shape)
      leftEye = shape[lStart:lEnd]
      rightEye = shape[rStart:rEnd]
      leftEAR = eye_aspect_ratio(leftEye)
      rightEAR = eye_aspect_ratio(rightEye)
      ear = (leftEAR + rightEAR) / 2.0
      leftEyeHull = cv2.convexHull(leftEye)
      rightEyeHull = cv2.convexHull(rightEye)
      cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
      cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
      if ear < EYE_AR_THRESH:
        COUNTER += 1
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
          if not ALARM_ON:
            ALARM_ON = True
            if args["alarm"] != "":
              t = Thread(target=sound_alarm,  args=(args["alarm"],))
              t.deamon = True
              t.start()
          cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          data['drowsy'] = 1
          print(f'${current_time} : Drowsy')
      else:
        data['drowsy'] = 0
        # data = {'drowsy': 0}
        COUNTER = 0
        ALARM_ON = False
      cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    json_data = json.dumps(data)
    await websocket.send(json_data)
    await asyncio.sleep(0.01)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break

  cv2.destroyAllWindows()
  vs.release()

start_server = websockets.serve(send_drowsiness_status, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
