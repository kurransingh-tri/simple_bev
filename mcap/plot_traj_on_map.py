import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np
from matplotlib.animation import FuncAnimation

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
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
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

def animate_trajectory(messages, window_seconds=2, hz=200):
    coords = parse_gnss_messages(messages)
    osm_tiles = cimgt.OSM()  # Use OpenStreetMap tiles

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=osm_tiles.crs)
    ax.set_extent([
        coords[:,1].min() - 0.001, coords[:,1].max() + 0.001,
        coords[:,0].min() - 0.001, coords[:,0].max() + 0.001
    ], crs=ccrs.PlateCarree())
    
    print("Adding OSM tiles...")
    ax.add_image(osm_tiles, 16)
    print("OSM tiles added.")

    traj_line, = ax.plot([], [], 'r-', linewidth=2, transform=ccrs.PlateCarree())
    point, = ax.plot([], [], 'bo', transform=ccrs.PlateCarree())

    window_size = int(window_seconds * hz)

    def init():
        traj_line.set_data([], [])
        point.set_data([], [])
        return traj_line, point

    def update(frame):
        start = max(0, frame - window_size)
        traj_line.set_data(coords[start:frame,1], coords[start:frame,0])
        point.set_data([coords[start,1]], [coords[start,0]])  # Trailing edge
        return traj_line, point

    ani = FuncAnimation(
        fig, update, frames=range(window_size, len(coords)),
        init_func=init, blit=True, interval=1
    )
    
    # Save animation as MP4 
    ani.save("trajectory_animation.mp4", writer="ffmpeg", fps=500)
    plt.show()

if __name__ == "__main__":
    messages = []
    for i in range(10):
        input_file = f"/home/ubuntu/lyftbags/tmpoc_tol/tmpoc_tol{i}.mcap"
        gnss_topic = "/novatel/gnsspos"

        messages_part_n = extract_gnss_messages(input_file, gnss_topic=gnss_topic)
        messages.extend(messages_part_n)
        print(f"Extracted {len(messages_part_n)} messages from {input_file}")

    # Import your animation function
    from plot_traj_on_map import animate_trajectory

    animate_trajectory(messages, window_seconds=2, hz=200)