import torch
import sys
sys.path.append(".")
from nets.bevformernet import Bevformernet
from nets.segnet import Segnet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import utils.vox
import utils.geom

# ImageNet normalization 
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # RGB
    std=[0.229, 0.224, 0.225]
)

class MCAPDataset(Dataset):
    def __init__(self, mcap_file, camera_ids, image_topic_fmt="/{}/image_compressed", calibration_topic='/calibration', target_size=(800, 448)):
        self.mcap_file = mcap_file
        self.camera_ids = camera_ids  # List of camera ids, e.g. ["cam0", "cam1"]
        self.image_topic_fmt = image_topic_fmt
        self.calibration_topic = calibration_topic
        self.target_size = target_size  # (W, H)
        self.calibrations = {}
        self.frames = self._load_frames_and_calibrations()

    def _load_frames_and_calibrations(self):
        # Load calibrations and all frames for each camera
        frames_dict = {cid: [] for cid in self.camera_ids}
        calibrations = {}
        with open(self.mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                # Calibration
                if channel.topic == self.calibration_topic:
                    for camera in proto_msg.cameras:
                        if camera.id in self.camera_ids:
                            calibrations[camera.id] = {
                                'description': camera.description,
                                'width': camera.width,
                                'height': camera.height,
                                'focal_length': {
                                    'x': camera.focal_length.x,
                                    'y': camera.focal_length.y
                                },
                                'optical_center': {
                                    'x': camera.optical_center.x,
                                    'y': camera.optical_center.y
                                },
                                'radial_distortion_coefs': list(camera.radial_distortion_coefs),
                                'pose': {
                                    'translation': {
                                        'x': camera.pose.translation.x,
                                        'y': camera.pose.translation.y,
                                        'z': camera.pose.translation.z
                                    },
                                    'rotation': {
                                        'row_1': list(camera.pose.rotation.row_1),
                                        'row_2': list(camera.pose.rotation.row_2),
                                        'row_3': list(camera.pose.rotation.row_3)
                                    }
                                }
                            }
                # Images
                for cid in self.camera_ids:
                    topic = self.image_topic_fmt.format(cid)
                    if channel.topic == topic:
                        frames_dict[cid].append({
                            'log_time': message.log_time,
                            'data': proto_msg.data
                        })
        self.calibrations = calibrations
        # Align frames by timestep 
        min_len = min(len(frames_dict[cid]) for cid in self.camera_ids)
        frames = []
        for i in range(min_len):
            frame = {cid: frames_dict[cid][i] for cid in self.camera_ids}
            frames.append(frame)
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # For each camera, get image, intrinsics, extrinsics
        images = []
        pix_T_cams = []
        cam0_T_cams = []
        timestamps = []
        for cid in self.camera_ids:
            frame = self.frames[idx][cid]
            calib = self.calibrations[cid]
            orig_W, orig_H = calib['width'], calib['height']
            # Intrinsics
            orig_intrinsics = np.array([
                [calib['focal_length']['x'], 0, calib['optical_center']['x']],
                [0, calib['focal_length']['y'], calib['optical_center']['y']],
                [0, 0, 1]
            ])
            sx = self.target_size[0] / float(orig_W)
            sy = self.target_size[1] / float(orig_H)
            intrinsics = orig_intrinsics.copy()
            intrinsics[0, 0] *= sx
            intrinsics[1, 1] *= sy
            intrinsics[0, 2] *= sx
            intrinsics[1, 2] *= sy
            # Extrinsics
            translation = np.array([
                calib['pose']['translation']['x'],
                calib['pose']['translation']['y'],
                calib['pose']['translation']['z']
            ])
            rotation = np.array([
                calib['pose']['rotation']['row_1'],
                calib['pose']['rotation']['row_2'],
                calib['pose']['rotation']['row_3']
            ])
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotation
            extrinsics[:3, 3] = translation
            # Image
            image_data = BytesIO(frame['data'])
            image = Image.open(image_data)
            image = image.resize(self.target_size, Image.BILINEAR)
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            images.append(image)
            pix_T_cam_4x4 = self.intrinsics_to_4x4(intrinsics)
            pix_T_cams.append(torch.from_numpy(pix_T_cam_4x4).float())
            cam0_T_cams.append(torch.from_numpy(extrinsics).float())
            timestamps.append(frame['log_time'])
        # Stack along S dimension
        images = torch.stack(images, dim=0)  # [S, C, H, W]
        pix_T_cams = torch.stack(pix_T_cams, dim=0)  # [S, 4, 4]
        cam0_T_cams = torch.stack(cam0_T_cams, dim=0)  # [S, 4, 4]
        return {
            'rgb_camXs': images,  # [S, C, H, W]
            'pix_T_cams': pix_T_cams,  # [S, 4, 4]
            'cam0_T_camXs': cam0_T_cams,  # [S, 4, 4]
            'timestamps': timestamps
        }

    def intrinsics_to_4x4(self, intrinsics_3x3):
        intrinsics_4x4 = np.eye(4)
        intrinsics_4x4[:3, :3] = intrinsics_3x3
        return intrinsics_4x4

def run_inference():
    # Configuration
    mcap_file = "/home/ubuntu/lyftbags/tol_06_09/tol_2.mcap"
    camera_ids = ["cam0", "cam4", "cam5", "cam6", "cam7", "cam8"]  # List all camera ids you want to use
    calibration_topic = "/calibration"
    checkpoint_path = "checkpoints/8x5_5e-4_rgb12_22:43:46/model-000025000.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Segnet(
        Z=200,  # BEV height dimension
        Y=8,  # BEV depth dimension
        X=200,  # BEV width dimension
        rand_flip=False,
        latent_dim=128,
        encoder_type="res101"
    ).to(device)

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()

    # Create dataset and dataloader
    dataset = MCAPDataset(mcap_file, camera_ids)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    scene_centroid_py = np.array([0.0, 1.0, 0.0]).reshape([1, 3])
    scene_centroid = torch.from_numpy(scene_centroid_py).float().to(device)
    XMIN, XMAX = -50, 50
    ZMIN, ZMAX = -50, 50
    YMIN, YMAX = -5, 5
    bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
    vox_util = utils.vox.Vox_util(
        Z=200,
        Y=8,
        X=200,
        scene_centroid=scene_centroid,
        bounds=bounds,
    )

    # Run inference
    with torch.no_grad():
        print("Starting inference...")
        print(f"Total frames to process: {len(dataset)}")
        for batch_idx, batch in enumerate(dataloader):
            # Keep batch dimension since model expects [B, S, C, H, W]
            rgb_camXs = batch['rgb_camXs'].to(device)  # [B, S, C, H, W]
            pix_T_cams = batch['pix_T_cams'].to(device)  # [B, S, 4, 4]
            cam0_T_camXs = batch['cam0_T_camXs'].to(device)  # [B, S, 4, 4]

            # Forward pass with batch dimension intact
            raw_e, feat_e, seg_e, center_e, offset_e = model(
                rgb_camXs,
                pix_T_cams,
                cam0_T_camXs,
                vox_util
            )
            print(f"seg_e shape: {seg_e.shape}")
            print(f"min seg_e: {seg_e.min()}, max seg_e: {seg_e.max()}")
            print(f"mean seg_e: {seg_e.mean()}, std seg_e: {seg_e.std()}")
            # # sigmoid and threshold
            # seg_e = torch.sigmoid(seg_e)
            # print(f"min seg_e: {seg_e.min()}, max seg_e: {seg_e.max()}")
            # print(f"mean seg_e: {seg_e.mean()}, std seg_e: {seg_e.std()}")
            # seg_e[seg_e > 0.5] = 1
            # seg_e[seg_e <= 0.5] = 0

            seg_np = seg_e[0][0].cpu().numpy() * 255  # shape (200, 200)
            cv2.imwrite(f'outputs/bev_seg_{batch_idx}.png', seg_np.astype(np.uint8))

            # Save a collage of all camera images in 2 rows (as balanced as possible)
            imgs = rgb_camXs[0].cpu().numpy()  # [S, C, H, W]
            imgs = np.transpose(imgs, (0, 2, 3, 1)) * 255  # [S, H, W, C]
            imgs = imgs.astype(np.uint8)
            S = imgs.shape[0]
            n_row1 = (S + 1) // 2
            n_row2 = S - n_row1
            row1_imgs = [cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR) for i in range(n_row1)]
            row2_imgs = [cv2.cvtColor(imgs[i + n_row1], cv2.COLOR_RGB2BGR) for i in range(n_row2)]
            row1 = np.concatenate(row1_imgs, axis=1) if row1_imgs else None
            row2 = np.concatenate(row2_imgs, axis=1) if row2_imgs else None
            if row2 is not None:
                # Pad row2 to match row1 width if needed
                if row2.shape[1] < row1.shape[1]:
                    pad_width = row1.shape[1] - row2.shape[1]
                    row2 = np.pad(row2, ((0,0),(0,pad_width),(0,0)), mode='constant', constant_values=0)
                collage = np.concatenate([row1, row2], axis=0)
            else:
                collage = row1
            cv2.imwrite(f'outputs/collage_{batch_idx}.png', collage)

            # Break after first frame for testing
            if batch_idx == 0:
                break

if __name__ == "__main__":
    run_inference()