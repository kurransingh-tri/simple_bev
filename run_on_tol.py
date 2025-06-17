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
    def __init__(self, mcap_file, image_topic, calibration_topic='/calibration', target_size=(800, 448)):  # W, H (note dim order switch from rest of code) 224*2, 400*2 for res_scale=2
        self.mcap_file = mcap_file
        self.image_topic = image_topic
        self.calibration_topic = calibration_topic
        self.calibrations = {}
        self.frames = self._load_frame_indices()
        self.target_size = target_size  # (W, H))
        
        with open(self.mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                if channel.topic == self.calibration_topic:
                    for camera in proto_msg.cameras:
                      self.calibrations[camera.id] = {
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
        # Original image size (from the camera)
        self.orig_W, self.orig_H = 1936, 1216 
        
        # Camera parameters (from draw_traj_on_img.py)
        self.orig_intrinsics = np.array([
            [2158.4365875475069, 0, 947.03386781214579],
            [0, 2150.1053705185745, 623.30766583992522],
            [0, 0, 1]
        ])
        
        # Scale intrinsics according to image scaling
        sx = self.target_size[1] / float(self.orig_W)  # scale_x = target_W / orig_W
        sy = self.target_size[0] / float(self.orig_H)  # scale_y = target_H / orig_H
        
        # Create scaled intrinsics
        self.intrinsics = self.orig_intrinsics.copy()
        self.intrinsics[0,0] *= sx  # fx
        self.intrinsics[1,1] *= sy  # fy
        self.intrinsics[0,2] *= sx  # cx
        self.intrinsics[1,2] *= sy  # cy

        
        translation = np.array([.24, 1.69, 1.57])
        rotation = np.array([
            [0.034738592286087, -0.99928182939666532, -0.015134584356187142],
            [0.005331648605275352, 0.01532881367816108, -0.999868291823663],
            [0.99938221103296132, 0.034653324643874273, 0.0058603209805749845]
        ])
        
        self.extrinsics = np.eye(4)
        self.extrinsics[:3, :3] = rotation
        self.extrinsics[:3, 3] = translation

    def _load_frame_indices(self):
        frames = []
        with open(self.mcap_file, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, proto_msg in reader.iter_decoded_messages():
                if channel.topic == self.image_topic:
                    frames.append({
                        'log_time': message.log_time,
                        'data': proto_msg.data
                    })
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image_data = BytesIO(frame['data'])
        image = Image.open(image_data)
        # Resize image to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # image = imagenet_normalize(image) # Normalize to ImageNet standards

        S = 1  # or whatever number of cameras your model expects
        image = image.unsqueeze(0).repeat(S, 1, 1, 1)  # [S, C, H, W]
        
        pix_T_cam = torch.from_numpy(self.intrinsics).float()
        pix_T_cam_4x4 = self.intrinsics_to_4x4(self.intrinsics)
        pix_T_cam = torch.from_numpy(pix_T_cam_4x4).float()


        cam0_T_cam = torch.from_numpy(self.extrinsics).float()
        return {
            'rgb_camXs': image,  # [S, C, H, W]
            'pix_T_cams': pix_T_cam.unsqueeze(0).repeat(S, 1, 1),  # [S, 4, 4]
            'cam0_T_camXs': cam0_T_cam.unsqueeze(0).repeat(S, 1, 1),  # [S, 4, 4]
            'timestamp': frame['log_time']
        }
        
    def intrinsics_to_4x4(self, intrinsics_3x3):
        intrinsics_4x4 = np.eye(4)
        intrinsics_4x4[:3, :3] = intrinsics_3x3
        return intrinsics_4x4

def run_inference():
    # Configuration
    mcap_file = "/home/ubuntu/lyftbags/tol_06_09/tol_2.mcap"
    image_topic = "/cam1/image_compressed"
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
    # print(f"All Keys: {model.state_dict().keys()}")    
    model.eval()

    # Create dataset and dataloader
    dataset = MCAPDataset(mcap_file, image_topic, calibration_topic)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    scene_centroid_py = np.array([0.0,
                                  1.0,
                                  0.0]).reshape([1, 3])
    scene_centroid = torch.from_numpy(scene_centroid_py).float().to(device)
    XMIN, XMAX = -50, 50
    ZMIN, ZMAX = -50, 50
    YMIN, YMAX = -5, 5
    bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
    # Initialize voxel utility
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
         # Iterate over the dataloader
         # Each batch contains one frame with RGB image and camera parameters
         # Assuming single camera input for simplicity
         # For multi-camera, you would need to adjust the dataset and model accordingly
        print(f"Total frames to process: {len(dataset)}")
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            rgb_camXs = batch['rgb_camXs'].to(device)  # [B,S,C,H,W]
            pix_T_cams = batch['pix_T_cams'].to(device)  # [B,S,4,4]
            cam0_T_camXs = batch['cam0_T_camXs'].to(device)  # [B,S,4,4]

            # Forward pass
            raw_e, feat_e, seg_e, center_e, offset_e = model(
                rgb_camXs,
                pix_T_cams,
                cam0_T_camXs,
                vox_util
            )

            # seg_output = torch.sigmoid(seg_e[0])  # shape (1, 200, 200)
            # seg_mask = (seg_output > 0.0).float()

            # # Save the mask as image
            # seg_np = seg_mask[0].cpu().numpy() * 255  # shape (200, 200), values in 0–255
            
            seg_np = seg_e[0][0].cpu().numpy() * 255  # shape (200, 200), values in 0–255
            cv2.imwrite(f'bev_seg_{batch_idx}.png', seg_np.astype(np.uint8))
            
            # Save the original image
            orig_img = batch['rgb_camXs'][0,0].cpu().numpy()  # [C, H, W]
            orig_img = np.transpose(orig_img, (1, 2, 0)) * 255  # [H, W, C], scale to 0-255
            orig_img = orig_img.astype(np.uint8)
            cv2.imwrite(f'orig_img_{batch_idx}.png', cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
            
            # Break after first frame for testing
            if batch_idx == 0:
                break

if __name__ == "__main__":
    run_inference()