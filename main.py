
import math
import shlex
import subprocess

import mediapipe as mp
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np
import time
import multiprocessing

from mediapipe.tasks.python.components.containers import NormalizedLandmark

from geom_helper import dist_sum
from hand2description import hand2detection, detection2description, actions2description

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

OPENAI_API_KEY = "[DELETED]"  # vsegpt.ru
OPENAI_API_BASE = "[DELETED]"
OPENAI_MODEL_TYPE = "openai/gpt-4"

model = ChatOpenAI(temperature=0.5, model=OPENAI_MODEL_TYPE, max_tokens=1024, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE)


if __name__ == '__main__':
    current_cell_selection = "left_up"
    permissions = {
        "oleg": ["left_down", "right_up", "right_down"],
        "luca": ["left_up", "left_down", "right_up", "right_down"],
        "maksim": ["left_up", "left_down", "right_up", "right_down"],
        "nikita": ["left_up", "left_down", "right_up", "right_down"],
        "vika": ["left_up", "left_down", "right_up", "right_down"]
    }

    img_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()

    gesture_model_path = 'models/gesture_recognizer.task'
    pose_model_path = 'models/pose_landmarker.task'
    face_model_path = 'models/blaze_face_short_range.tflite'
    hand_model_path = 'models/hand_landmarker.task'

    delegate_option = 1

    gesture_base_options = python.BaseOptions(model_asset_path=gesture_model_path, delegate=delegate_option)
    gesture_options = vision.GestureRecognizerOptions(
        base_options=gesture_base_options,
        num_hands=10)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path, delegate=delegate_option)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        output_segmentation_masks=False,
        num_poses=2)
    pose_recognizer = vision.PoseLandmarker.create_from_options(pose_options)

    face_base_options = python.BaseOptions(model_asset_path=face_model_path, delegate=delegate_option)
    face_options = vision.FaceDetectorOptions(
        base_options=face_base_options)
    face_recognizer = vision.FaceDetector.create_from_options(face_options)

    hand_base_options = python.BaseOptions(model_asset_path=hand_model_path, delegate=delegate_option)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2)
    hand_recognizer = vision.HandLandmarker.create_from_options(hand_options)

    def draw_hand_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cnt = 0
    pose_freq = 1
    last_trigger = time.time()

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = frame[..., ::-1].copy()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = gesture_recognizer.recognize(img)
    detection_result = pose_recognizer.detect(img)

    hand_landmark_buffer = []
    hand_landmark_diff_buffer = []

    hand_action_state_buffer = []

    hand_state = {
        'left': {'static': True, 'speed': 0},
        'right': {'static': True, 'speed': 0}
    }

    def stub(*args, **kwargs):
        pass

    def handle_image(image, *args, **kwargs):
        global hand_action_state_buffer
        hand_recognizer_result = hand_recognizer.detect(image)
        hand_detection = hand2detection(hand_recognizer_result)
        #hand_description = detection2description(hand_detection.get('right', hand_detection.get('left', {'fingers': {}})))
        hand_description = actions2description(hand_action_state_buffer, hand_landmark_diff_buffer)
        print(hand_description)
        messages = [
            SystemMessage(content="""I can describe any type of hand gesture, based on information about fingers position.
            If gesture consist of different static hand positions and trajectory of hand between them, I need to describe whole gesture at once.
            I keep my answers short and precise, stating short name of the gesture, its meaning and possible task for robot.
            Robot is capable of taking objects, moving around etc
            Example of answer:
             
            Name of gesture: Fist
            Meaning of gesture: Grabbing something or threatening somebody
            Task: Take an object you are looking now"""),
            HumanMessage(content='This is description of fingers of hand, please say what is the gesture here described:\n' + hand_description),
        ]
        res = model.invoke(messages)
        print("------------------------------")
        print(res.content)
        print("------------------------------")
        return res.content


    handle_image_start_state = 'hand_opened'

    def handle_image_start(*args, **kwargs):
        global handle_image_start_state
        global hand_action_state_buffer
        global hand_state
        if handle_image_start_state == 'hand_opened':
            handle_image_start_state = 'hand_closed'
            hand_action_state_buffer = []
            hand_state = {
                'left': {'static': True, 'speed': 0},
                'right': {'static': True, 'speed': 0}
            }

    def handle_image_end(*args, **kwargs):
        global handle_image_start_state
        global hand_action_state_buffer
        global hand_state
        if handle_image_start_state == 'hand_closed':
            handle_image_start_state = 'hand_opened'
            handle_image(*args, **kwargs)

    gesture_mapping = {
        ("Left", "Open_Palm"): handle_image_end,
        ("Right", "Open_Palm"): stub,
        ("Left", "Closed_Fist"): handle_image_start,
        ("Right", "Closed_Fist"): stub
    }

    gesture_trigger_delay = 1

    face_detection_window_size = 7
    prev_detection_window = [set() for i in range(face_detection_window_size)]
    face_permission_tags = set()

    hand_diff_eps = 0.2
    hand_state = {
        'left': {'static': True, 'speed': 0},
        'right': {'static': True, 'speed': 0}
    }

    while True:
        cnt += 1
        start_point = time.time()
        hand_data = {
            "Left": {},
            "Right": {}
        }
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        scale_percent = 25  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        frame = frame[..., ::-1].copy()
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        resized_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
        gesture_recognition_result = gesture_recognizer.recognize(img)
        #hand_detection_result = hand_recognizer.detect(img)

        new_buf_elem = {
            'right': None,
            'left': None
        }
        new_diff_elem = {
            'right': None,
            'left': None
        }

        for i in range(len(gesture_recognition_result.hand_landmarks)):
            landmarks = gesture_recognition_result.hand_landmarks[i]
            handedness = gesture_recognition_result.handedness[i]
            hand_type = 'right' if handedness[0].category_name == 'Right' else 'left'
            new_buf_elem[hand_type] = landmarks

            diff = []
            for i in range(len(landmarks)):
                new_land_elem = NormalizedLandmark(x=0, y=0, z=0)
                if hand_landmark_buffer and hand_landmark_buffer[-1][hand_type] is not None:
                    x_diff = hand_landmark_buffer[-1][hand_type][i].x - landmarks[i].x
                    y_diff = hand_landmark_buffer[-1][hand_type][i].y - landmarks[i].y
                    z_diff = hand_landmark_buffer[-1][hand_type][i].z - landmarks[i].z
                    new_land_elem = NormalizedLandmark(x=x_diff, y=y_diff, z=z_diff)
                diff.append(new_land_elem)
            new_diff_elem[hand_type] = diff

            new_diff_right = dist_sum(new_diff_elem[hand_type])
            if new_diff_right > hand_diff_eps:
                if hand_state[hand_type]['static'] and hand_type == 'right':
                    hand_action_state_buffer.append({
                        'static': True,
                        'landmarks': landmarks,
                        'state_index': max(len(new_buf_elem) - 1, 0)
                    })
                hand_state[hand_type]['static'] = False
                hand_state[hand_type]['speed'] = new_diff_right
            else:
                if not hand_state[hand_type]['static'] and hand_type == 'right':
                    hand_action_state_buffer.append({
                        'static': False,
                        'landmarks': landmarks,
                        'state_index': max(len(new_buf_elem) - 1, 0)
                    })
                hand_state[hand_type]['static'] = True
                hand_state[hand_type]['speed'] = new_diff_right

        hand_landmark_buffer.append(new_buf_elem)
        hand_landmark_diff_buffer.append(new_diff_elem)

        if not hand_state['right']['static']:
            print("not static", hand_state['right']['speed'])

        """
        print("-------------------------")
        hd = hand2detection(hand_detection_result)
        for hand in hd:
            if hd[hand]:
                for finger in hd[hand]['fingers']:
                    if hd[hand]['fingers'][finger]['in_fist']:
                        print("Finger", finger, "on", hand, "hand is in fist")"""

        if cnt % pose_freq == 0:
            detection_result = pose_recognizer.detect(resized_img)
        for gesture, hand_info in zip(gesture_recognition_result.gestures, gesture_recognition_result.handedness):
            gesture_info = set([gesture[0].category_name])
            hand_data[hand_info[0].category_name] = gesture_info
        if img_queue.empty():
            img_queue.put(frame)
        try:
            data = res_queue.get_nowait()
            prev_detection = set()
            for detection in prev_detection_window:
                prev_detection = prev_detection.union(detection)
            prev_detection_window = [tag_set for i, tag_set in enumerate(prev_detection_window) if i != 0]
            current_detection = set()

            prev_detection_window.append(current_detection)
            current_detection = set()
            for detection in prev_detection_window:
                current_detection = current_detection.union(detection)
            face_permission_tags = current_detection
            if prev_detection != current_detection:
                # print(prev_detection, current_detection)
                for tag in list(prev_detection.difference(current_detection)):
                    print(tag, "out of frame")
                for tag in list(current_detection.difference(prev_detection)):
                    print(tag, "now in frame")
                    command = "vlc voices/" + tag + ".mp3"
                    proc = subprocess.Popen(
                        shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
                    )
        except Exception:
            pass
        if time.time() - last_trigger > gesture_trigger_delay:
            for data in hand_data["Right"]:
                print(data, "right")
                gesture_mapping.get(("Right", data), stub)(img, face_permission_tags, permissions, current_cell_selection)
                last_trigger = time.time()
            for data in hand_data["Left"]:
                print(data, "left")
                gesture_mapping.get(("Left", data), stub)(img, face_permission_tags, permissions, current_cell_selection)
                last_trigger = time.time()
        if len(detection_result.pose_landmarks):
            elbow = detection_result.pose_landmarks[0][14]
            wrist = detection_result.pose_landmarks[0][16]
            diff_x = math.atan((elbow.x - wrist.x) * 1000)
            diff_y = math.atan((elbow.y - wrist.y) * 1000)
            if elbow.presence > 0.75 and wrist.presence > 0.75:
                if diff_x < 0 and diff_y > 0:
                    current_cell_selection = "right_up"
                    print("right_up")
                if diff_x < 0 and diff_y < 0:
                    current_cell_selection = "right_down"
                    print("right_down")
                if diff_x > 0 and diff_y > 0:
                    current_cell_selection = "left_up"
                    print("left_up")
                if diff_x > 0 and diff_y < 0:
                    current_cell_selection = "left_down"
                    print("left_down")
        #print(time.time() - start_point)
        annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
        annotated_image = draw_hand_landmarks_on_image(annotated_image, gesture_recognition_result)
        annotated_image = annotated_image[..., ::-1].copy()
        cv2.imshow('frame', annotated_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()