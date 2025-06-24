import numpy as np
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

def rad_to_deg(rad):
    return np.degrees(rad)

def parse_gnss_messages(messages):
    """
    messages: list of dicts with keys 'lat_rad', 'lng_rad', 'height_msl_m'
    Returns: np.array of shape (N, 2) with lat, lon in degrees
    """
    lats = [rad_to_deg(msg['lat_rad']) for msg in messages]
    lons = [rad_to_deg(msg['lng_rad']) for msg in messages]
    return np.array(list(zip(lats, lons)))

def extract_gnss_messages(input_file, gnss_topic="/novatel/gnsspos"):
    messages = []
    with open(input_file, "rb") as infile:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            if channel.topic == gnss_topic:
                msg = {
                    "lat_rad": getattr(proto_msg, "lat_rad", None),
                    "lng_rad": getattr(proto_msg, "lng_rad", None),
                    "height_msl_m": getattr(proto_msg, "height_msl_m", None)
                }
                if None not in msg.values():
                    messages.append(msg)
    return messages

def extract_localization_messages(input_file, localization_topic="/localization/motion_state_local"):
    """
    Sample message structure:
    Message data: timestamped_isometry {
    timestamp_ns: 1433864546395926000
    isometry3d {
        translation {
        x: 138.64453797719617
        y: 578.0308137931562
        z: -3.0412877543038435
        }
        rotation {
        row_1: -0.99979995619899675
        row_1: 0.014294682039156081
        row_1: -0.013989626509793499
        row_2: -0.013757273910169281
        row_2: -0.99919118734718793
        row_2: -0.037785030663970376
        row_3: -0.014518436522045484
        row_3: -0.037585012879019136
        row_3: 0.99918796120041364
        }
    }
    }
    linear_velocity_mps {
    x: -30.291305297506142
    y: -0.40071158079725716
    z: -0.082488131781168428
    }
    angular_velocity_radps {
    x: 0.0012891836087127163
    y: -0.027008286397320315
    z: 0.0090365709215241
    }
    linear_acceleration_mps2 {
    x: 0.77117632830588734
    y: -0.42498870150212803
    z: 0.33008305217244649
    }
    motion_state_source: MOTION_STATE_SOURCE_LOCAL_SLAM

    """
    messages = []
    with open(input_file, "rb") as infile:
        reader = make_reader(infile, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            if channel.topic == localization_topic:
                tsi = getattr(proto_msg, "timestamped_isometry", None)
                if tsi is not None:
                    isometry = getattr(tsi, "isometry3d", None)
                    if isometry is not None:
                        msg = {
                            "timestamp_ns": getattr(tsi, "timestamp_ns", None),
                            "translation": getattr(isometry, "translation", None),
                            "rotation": getattr(isometry, "rotation", None)
                        }
                        if msg["translation"] is not None and msg["rotation"] is not None:
                            messages.append(msg)
    return messages

def write_slam_tum(messages, output_file, start_idx=0, mode='w'):
    """
    messages: list of dicts with keys 'translation', 'rotation'
    Each 'translation' has x, y, z.
    Each 'rotation' is a 3x3 matrix in row-major order.
    start_idx: starting index for timestamps
    mode: file open mode ('w' for write, 'a' for append)
    Returns: last index written + 1
    """
    with open(output_file, mode) as f:
        for i, msg in enumerate(messages):
            t = msg['translation']
            r = msg['rotation']
            rot_matrix = np.array([
                [getattr(r, 'row_1', [None]*3)[0], getattr(r, 'row_1', [None]*3)[1], getattr(r, 'row_1', [None]*3)[2]],
                [getattr(r, 'row_2', [None]*3)[0], getattr(r, 'row_2', [None]*3)[1], getattr(r, 'row_2', [None]*3)[2]],
                [getattr(r, 'row_3', [None]*3)[0], getattr(r, 'row_3', [None]*3)[1], getattr(r, 'row_3', [None]*3)[2]],
            ])
            quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
            timestamp = start_idx + i
            f.write(f"{timestamp} {t.x} {t.y} {t.z} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")
    return start_idx + len(messages)

def write_gnss_tum(messages, output_file, start_idx=0, mode='w'):
    """
    messages: list of dicts with keys 'lat_rad', 'lng_rad', 'height_msl_m'
    Writes to TUM format with identity quaternion.
    start_idx: starting index for timestamps
    mode: file open mode ('w' for write, 'a' for append)
    Returns: last index written + 1
    """
    with open(output_file, mode) as f:
        for i, msg in enumerate(messages):
            t = start_idx + i
            x = np.degrees(msg['lat_rad'])
            y = np.degrees(msg['lng_rad'])
            z = msg['height_msl_m']
            qx, qy, qz, qw = 0, 0, 0, 1
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
    return start_idx + len(messages)

if __name__ == "__main__":
    slam_output_file = "/home/ubuntu/simple_bev/mcap/trajectory_slam.tum"
    gnss_output_file = "/home/ubuntu/simple_bev/mcap/trajectory_gnss.tum"
    last_idx = 0
    last_gnss_idx = 0

    for i in range(11):
        print(f"Processing file: tmpoc_tol{i}.mcap")
        input_file = f"/home/ubuntu/lyftbags/tmpoc_tol/tmpoc_tol{i}.mcap"
        slam_messages = extract_localization_messages(input_file)
        gnss_messages = extract_gnss_messages(input_file)
        print(f"Found {len(slam_messages)} slam messages, {len(gnss_messages)} gnss messages")
        mode = 'w' if i == 0 else 'a'
        last_idx = write_slam_tum(slam_messages, slam_output_file, start_idx=last_idx, mode=mode)
        last_gnss_idx = write_gnss_tum(gnss_messages, gnss_output_file, start_idx=last_gnss_idx, mode=mode)

