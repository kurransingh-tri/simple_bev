import cv2
import numpy as np
import json
from io import BytesIO
from PIL import Image

import mcap
from mcap_protobuf.decoder import DecoderFactory
from mcap.reader import make_reader
from mcap_protobuf.writer import Writer

def parse_calibration_data():
    """
    Parses the provided camera calibration data and returns intrinsics and extrinsics.
    """
    # Intrinsics
    focal_length_x = 2158.4365875475069
    focal_length_y = 2150.1053705185745
    optical_center_x = 947.03386781214579
    optical_center_y = 623.30766583992522

    intrinsics = np.array([
        [focal_length_x, 0, optical_center_x],
        [0, focal_length_y, optical_center_y],
        [0, 0, 1]
    ])

    # Extrinsics (pose)
    # translation = np.array([1.6392844988916322, 0.24175588834906847, 1.5727169645660566])
    translation = np.array([.24, 1.69, 1.57])

    # translation = np.array([0.0, 0.0, 0.0])
    rotation = np.array([
        [0.034738592286087, -0.99928182939666532, -0.015134584356187142],
        [0.005331648605275352, 0.01532881367816108, -0.999868291823663],
        [0.99938221103296132, 0.034653324643874273, 0.0058603209805749845]
    ])

    # Combine rotation and translation into a 4x4 transformation matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation

    return intrinsics, extrinsics


def process_mcap(input_file, image_topic, calibration_topic, trajectories):
    """
    Reads an MCAP file, projects trajectories onto images, and writes modified images to a new MCAP file.
    """
    with open(input_file, "rb") as infile:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])


        # Parse calibration data
        intrinsics, extrinsics = parse_calibration_data()

        message_counts = {}
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            # Initialize message count for each topic
            if channel.topic not in message_counts:
                message_counts[channel.topic] = 0
            message_counts[channel.topic] += 1
            # print(f"Processing message on topic: {channel.topic}, log_time: {message.log_time}")
            # if channel.topic == image_topic:
            #     # Convert the byte data to a file-like object
            #     image_data = BytesIO(proto_msg.data)
            #     # Load the image using PIL
            #     im = Image.open(image_data)
            #     # Decode image
            #     image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            #     cv2.imshow("Modified Image", image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #     _, encoded_image = cv2.imencode(".jpg", image)
            #     message_count += 1
            #     break
            # if channel.topic.startswith("/arene/perception"):
            #     print(f"Processing message on topic: {channel.topic}, log_time: {message.log_time}")
            #     print(f"Message data: {message.data}")
            if channel.topic == "/novatel/gnsspos":
                print("Message data:", proto_msg)
        
if __name__ == "__main__":
    input_file = "/home/ubuntu/lyftbags/tmpoc_tol/tmpoc_tol2.mcap"
    image_topic = "/cam0/image_compressed"
    calibration_topic = "/calibration"

    # Example trajectories (replace with actual data)
    trajectories = [
        {"timestamp": 0, "x": 0, "y": 0, "z": 0},
        {"timestamp": 1, "x": 1, "y": 1, "z": 1},
        # Add more trajectory points as needed
    ]

    process_mcap(input_file, image_topic, calibration_topic, trajectories)