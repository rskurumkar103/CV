# import cv2
# import numpy as np
# import streamlit as st
# from twilio.rest import Client
# from geopy.geocoders import Nominatim
# from PIL import Image
# import time

# # Twilio configuration
# account_sid = 'ACdb98db67e53a47c1eb6876b047818ac5'
# auth_token = 'b05e2e4c9658751f6482f3602cd051ed'
# twilio_phone_number = '+12182559854'
# your_phone_number = '+919767161003'
# client = Client(account_sid, auth_token)

# # Initialize geocoder
# geolocator = Nominatim(user_agent="weapon_detection")

# # Load Yolo model
# net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
# classes = ["Weapon"]
# output_layer_names = net.getUnconnectedOutLayersNames()
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# # Get current location
# def get_current_location():
#     location = geolocator.geocode("Me")
#     if location:
#         return location.latitude, location.longitude
#     else:
#         return None

# # Send location via Twilio
# def send_location(latitude, longitude):
#     message = client.messages.create(
#         body=f'Weapon detected at location: Latitude - {latitude}, Longitude - {longitude}',
#         from_=twilio_phone_number,
#         to=your_phone_number
#     )
#     st.success("Alert message sent with current location!")

# # Streamlit app
# st.title("Weapon Detection System")
# st.sidebar.header("Settings")

# # Input file or webcam
# input_source = st.sidebar.radio(
#     "Select Input Source",
#     ("Webcam", "Upload Video")
# )

# # Display video feed
# if input_source == "Webcam":
#     cap = cv2.VideoCapture(0)
# else:
#     uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         cap = cv2.VideoCapture(uploaded_file.name)

# if st.button("Start Detection"):
#     if not cap.isOpened():
#         st.error("Error: Unable to open video source.")
#     else:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("Video ended or no frames available.")
#                 break

#             height, width, _ = frame.shape
#             blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#             net.setInput(blob)
#             outs = net.forward(output_layer_names)

#             class_ids = []
#             confidences = []
#             boxes = []
#             for out in outs:
#                 for detection in out:
#                     scores = detection[5:]
#                     class_id = np.argmax(scores)
#                     confidence = scores[class_id]
#                     if confidence > 0.5 and class_id == 0:  # Weapon detected
#                         center_x = int(detection[0] * width)
#                         center_y = int(detection[1] * height)
#                         w = int(detection[2] * width)
#                         h = int(detection[3] * height)
#                         x = int(center_x - w / 2)
#                         y = int(center_y - h / 2)
#                         boxes.append([x, y, w, h])
#                         confidences.append(float(confidence))
#                         class_ids.append(class_id)

#             indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#             for i in range(len(boxes)):
#                 if i in indexes:
#                     x, y, w, h = boxes[i]
#                     label = str(classes[class_ids[i]])
#                     color = colors[class_ids[i]]
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                     cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

#                     # Trigger alarm and send location
#                     location = get_current_location()
#                     if location:
#                         send_location(location[0], location[1])

#             # Convert frame to display in Streamlit
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             st.image(frame, channels="RGB")
#             time.sleep(0.03)

#         cap.release()


import cv2
import numpy as np
import streamlit as st
from twilio.rest import Client
from geopy.geocoders import Nominatim
import time

# Twilio configuration
account_sid = 'ACdb98db67e53a47c1eb6876b047818ac5'
auth_token = 'b05e2e4c9658751f6482f3602cd051ed'
twilio_phone_number = '+12182559854'
your_phone_number = '+919767161003'
client = Client(account_sid, auth_token)

# Initialize geocoder
geolocator = Nominatim(user_agent="weapon_detection")

# Load Yolo model
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Get current location
def get_current_location():
    location = geolocator.geocode("Me")
    if location:
        return location.latitude, location.longitude
    else:
        return None

# Send location via Twilio
def send_location(latitude, longitude):
    message = client.messages.create(
        body=f'Weapon detected at location: Latitude - {latitude}, Longitude - {longitude}',
        from_=twilio_phone_number,
        to=your_phone_number
    )
    st.success("Alert message sent with current location!")

# Streamlit app
st.title("Weapon Detection System")
st.sidebar.header("Settings")

# Input file or webcam
input_source = st.sidebar.radio(
    "Select Input Source",
    ("Webcam", "Upload Video")
)

# Display video feed
if input_source == "Webcam":
    cap = cv2.VideoCapture(0)
else:
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        cap = cv2.VideoCapture(uploaded_file.name)

if st.button("Start Detection"):
    if not cap.isOpened():
        st.error("Error: Unable to open video source.")
    else:
        # Placeholder for video frame
        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Video ended or no frames available.")
                break

            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layer_names)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:  # Weapon detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    # Trigger alarm and send location
                    location = get_current_location()
                    if location:
                        send_location(location[0], location[1])

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the single placeholder with the latest frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Add a slight delay for frame updates
            time.sleep(0.03)

        cap.release()
