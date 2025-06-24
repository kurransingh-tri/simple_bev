from pyproj import Transformer
## IMPORTANT: note that lat and lon gets swapped in the transform function as tum format is x, y (lon, lat)

# WGS84 (lat/lon) → UTM Zone 17N (covers Ohio, based on your lat/lon)
transformer = Transformer.from_crs("epsg:4326", "epsg:32617", always_xy=True)

with open("mcap/trajectory_gnss.tum") as f_in, open("mcap/trajectory_gnss_utm.tum", "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        t, lat, lon, alt, qx, qy, qz, qw = parts
        x, y = transformer.transform(float(lon), float(lat))
        f_out.write(f"{t} {x} {y} {alt} {qx} {qy} {qz} {qw}\n")
