"""
This script reads an MCAP file containing images and camera calibration data,
loads trajectories from a JSON file, and projects the trajectories onto the images.
It then writes the modified images with projected trajectories to a new MCAP file.
"""

import cv2
import numpy as np
import json
from io import BytesIO
from PIL import Image

import mcap
from mcap_protobuf.decoder import DecoderFactory
from mcap.reader import make_reader
from mcap_protobuf.writer import Writer

def load_diffusion_trajectories(json_file):
    """
    Loads diffusion trajectories from a JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract all trajectories for just first frame
    frame = data[1]
    trajectories = frame["paths"]
    
    # lets only visualize denoising for first trajectory
    trajectories = trajectories[5]
    print(f"Loaded {len(trajectories)} trajectories from {json_file}")
    print(np.array(trajectories).shape)
    return trajectories
    
    

def load_trajectories_from_json(json_file):
    """
    Loads trajectories from a JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract paths from each frame
    trajectories = {}
    for frame in data:
        trajectories[frame["frame_index"]] = frame["paths"]
    # print(f"Loaded {len(trajectories)} trajectories from {json_file}")
    # print(np.array(trajectories[0]).shape)
    return trajectories

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

def project_points_to_image(points, extrinsics, intrinsics):
    """
    Projects 3D points onto the image plane using camera extrinsics and intrinsics.
    """
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply extrinsics (world to camera transformation)
    camera_coords = extrinsics @ points_homogeneous.T
    # Apply intrinsics (camera to image transformation)
    image_coords = intrinsics @ camera_coords[:3, :]
    # Normalize to get pixel coordinates
    image_coords /= image_coords[2, :]
    return image_coords[:2, :].T

def process_mcap(input_file, output_file, image_topic, calibration_topic, trajectories):
    """
    Reads an MCAP file, projects trajectories onto images, and writes modified images to a new MCAP file.
    """
    with open(input_file, "rb") as infile, open(output_file, "wb") as outfile, Writer(outfile) as writer:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])


        # Parse calibration data
        intrinsics, extrinsics = parse_calibration_data()

        message_count = 0
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            if channel.topic == image_topic:
                # first 5 seconds are not in the predicted trajectories 
                if message_count < 40:
                    message_count += 1
                    continue
                # Convert the byte data to a file-like object
                image_data = BytesIO(proto_msg.data)
                # Load the image using PIL
                im = Image.open(image_data)
                # Decode image
                image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

                # Project trajectories onto the image
                points = np.array([[x, y, 0] for x, y in trajectories[message_count][0]])
                # points = np.array([[1,0,0],[5,0,0],[10,0,0],[15,0,0],[20,0,0]])
                projected_points = project_points_to_image(points, extrinsics, intrinsics)
                for x, y in projected_points:
                    cv2.circle(image, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
                # Encode modified image
                cv2.imshow("Modified Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                _, encoded_image = cv2.imencode(".jpg", image)
                # Write modified image to the new MCAP file
                # writer.write_message(channel.topic + "_traj", encoded_image.tobytes(), message.log_time)
                message_count += 1
                break
        writer.finish()
        
def process_mcap_to_video_single_traj(input_file, output_video, image_topic, calibration_topic, trajectories):
    """
    Reads an MCAP file, projects trajectories onto images, and saves the frames to an MP4 video file.
    """
    with open(input_file, "rb") as infile:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])

        # Parse calibration data
        intrinsics, extrinsics = parse_calibration_data()

        # Initialize the video writer
        video_writer = None
        frame_width, frame_height = None, None
        fps = 10  # Set the desired frames per second for the video

        skip_frames = 30
        message_count = 0
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            if channel.topic == image_topic:
                # # First 5 seconds are not in the predicted trajectories
                if message_count < skip_frames:
                    message_count += 1
                    continue

                # Convert the byte data to a file-like object
                image_data = BytesIO(proto_msg.data)
                # Load the image using PIL
                im = Image.open(image_data)
                # Decode image
                image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

                # Initialize the video writer if not already initialized
                if video_writer is None:
                    frame_height, frame_width = image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'mp4v' codec for MP4 files
                        fps,
                        (frame_width, frame_height)
                    )

                # Project trajectories onto the image
                if message_count - skip_frames not in trajectories:
                    break
                points = np.array([[x, y, 0] for x, y in trajectories[message_count-skip_frames][0]])
                projected_points = project_points_to_image(points, extrinsics, intrinsics)
                for x, y in projected_points:
                    cv2.circle(image, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

                # Write the frame to the video
                video_writer.write(image)

                message_count += 1

        # Release the video writer
        if video_writer is not None:
            video_writer.release()
            
def process_mcap_to_video_multi_traj(input_file, output_video, image_topic, calibration_topic, trajectories):
    """
    Reads an MCAP file, projects trajectories onto images, and saves the frames to an MP4 video file.
    """
    with open(input_file, "rb") as infile:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])

        # Parse calibration data
        intrinsics, extrinsics = parse_calibration_data()

        # Initialize the video writer
        video_writer = None
        frame_width, frame_height = None, None
        fps = 10  # Set the desired frames per second for the video

        skip_frames = 30
        message_count = 0

        # Predefine a consistent color for each trajectory index
        max_trajectories = max(len(frame) for frame in trajectories.values())
        trajectory_colors = {
            index: tuple(np.random.randint(0, 256, size=3).tolist())
            for index in range(max_trajectories)
        }

        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            if channel.topic == image_topic:
                # Skip the first few frames
                if message_count < skip_frames:
                    message_count += 1
                    continue

                # Convert the byte data to a file-like object
                image_data = BytesIO(proto_msg.data)
                # Load the image using PIL
                im = Image.open(image_data)
                # Decode image
                image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

                # Initialize the video writer if not already initialized
                if video_writer is None:
                    frame_height, frame_width = image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'mp4v' codec for MP4 files
                        fps,
                        (frame_width, frame_height)
                    )

                # Project trajectories onto the image
                if message_count - skip_frames not in trajectories:
                    break

                # Loop through all trajectories for the current frame
                for trajectory_index, trajectory in enumerate(trajectories[message_count - skip_frames]):
                    # Get the predefined color for this trajectory index
                    color = trajectory_colors.get(trajectory_index, (255, 255, 255))  # Default to white if not found

                    # Convert trajectory points to 3D points
                    points = np.array([[x, y, 0] for x, y in trajectory])

                    # Project points onto the image
                    projected_points = project_points_to_image(points, extrinsics, intrinsics)

                    # Draw the trajectory on the image
                    for x, y in projected_points:
                        cv2.circle(image, (int(x), int(y)), radius=7, color=color, thickness=-1)

                # Write the frame to the video
                video_writer.write(image)

                message_count += 1

        # Release the video writer
        if video_writer is not None:
            video_writer.release()

def process_mcap_to_video_diffusion(input_file, output_video, image_topic, calibration_topic, trajectories):
    """
    Reads an MCAP file, projects denoising process onto first frame, and saves the video to an MP4 file.
    """
    print(np.array(trajectories).shape)
    with open(input_file, "rb") as infile:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])

        # Parse calibration data
        intrinsics, extrinsics = parse_calibration_data()

        # Initialize the video writer
        video_writer = None
        frame_width, frame_height = None, None
        fps = 30  # Set the desired frames per second for the video

        skip_frames = 30
        message_count = 0
        
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            if channel.topic == image_topic:
                # Skip the first few frames
                if message_count < skip_frames:
                    message_count += 1
                    continue

                # Convert the byte data to a file-like object
                image_data = BytesIO(proto_msg.data)
                # Load the image using PIL
                im = Image.open(image_data)
                # Decode image
                image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

                # Initialize the video writer if not already initialized
                if video_writer is None:
                    frame_height, frame_width = image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),  # Use 'mp4v' codec for MP4 files
                        fps,
                        (frame_width, frame_height)
                    )

                original_image = image.copy()

                # Project denoising onto the image
                # Loop through all trajectories for the current frame
                for traj_index, traj in enumerate(trajectories):
                    # rest image to original clean version
                    image = original_image.copy()
                    
                    # Get the predefined color for this trajectory index
                    color = (0, 0, 255)  # Default to white if not found

                    # Convert trajectory points to 3D points
                    points = np.array([[x, y, 0] for x, y in traj])

                    # Project points onto the image
                    projected_points = project_points_to_image(points, extrinsics, intrinsics)
                    
                    # Draw the trajectory on the image
                    for x, y in projected_points:
                        cv2.circle(image, (int(x), int(y)), radius=10, color=color, thickness=-1)

                    # Write the frame to the video
                    video_writer.write(image)

                break

        # Release the video writer
        if video_writer is not None:
            video_writer.release() 
            
def main():
    input_mcap = "cam0_ole.mcap"
    output_mcap = "cam0_ole_paths.mcap"
    image_topic = "/cam0/image_compressed"
    calibration_topic = "/calibration"
    json_file = "ole_simulation_0_scenename_tri401_1716326717000_1716326739012_missionid_10144194590685643208_9607670596552160349_path.json"
    trajectories = load_trajectories_from_json(json_file)
    
    diffusion_json_file = "ole_simulation_0_scenename_tri401_1716326717000_1716326739012_missionid_10144194590685643208_9607670596552160349_diffusion_denoising.json"
    diffusion_trajectories = load_diffusion_trajectories(diffusion_json_file)
    
    output_video = "cam0_ole_diffusion.mp4"

    process_mcap_to_video_diffusion(input_mcap, output_video, image_topic, calibration_topic, diffusion_trajectories)
    # process_mcap_to_video_multi_traj(input_mcap, output_video, image_topic, calibration_topic, trajectories)    
    # process_mcap_to_video_single_traj(input_mcap, output_video, image_topic, calibration_topic, trajectories)
    # process_mcap(input_mcap, output_mcap, image_topic, calibration_topic, trajectories)

if __name__ == "__main__":
    main()

