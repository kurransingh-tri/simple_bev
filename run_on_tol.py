import torch
import sys
sys.path.append(".")
from nets.bevformernet import Bevformernet
from torch.utils.data import Dataset, DataLoader
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import utils.vox
import utils.geom

class MCAPDataset(Dataset):
    def __init__(self, mcap_file, image_topic):
        self.mcap_file = mcap_file
        self.image_topic = image_topic
        self.frames = self._load_frame_indices()
        
        # Camera parameters (from draw_traj_on_img.py)
        self.intrinsics = np.array([
            [2158.4365875475069, 0, 947.03386781214579],
            [0, 2150.1053705185745, 623.30766583992522],
            [0, 0, 1]
        ])
        
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
        
        # Convert bytes to image
        image_data = BytesIO(frame['data'])
        image = Image.open(image_data)
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to tensor and add batch dimension for single camera
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)  # Add camera dimension [1,C,H,W]
        
        # Convert camera parameters to tensors
        pix_T_cam = torch.from_numpy(self.intrinsics).float()
        cam0_T_cam = torch.from_numpy(self.extrinsics).float()
        
        return {
            'rgb_camXs': image,  # [1,C,H,W]
            'pix_T_cams': pix_T_cam.unsqueeze(0),  # [1,4,4]
            'cam0_T_camXs': cam0_T_cam.unsqueeze(0),  # [1,4,4]
            'timestamp': frame['log_time']
        }

def run_inference():
    # Configuration
    mcap_file = "cam0_ole.mcap"
    image_topic = "/cam0/image_compressed"
    checkpoint_path = "checkpoints/8x5_5e-4_rgb12_22:43:46/model-000025000.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Bevformernet(
        Z=200,  # BEV height dimension
        Y=200,  # BEV depth dimension
        X=200,  # BEV width dimension
        rand_flip=False,
        latent_dim=128,
        encoder_type="res101"
    ).to(device)

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dataset and dataloader
    dataset = MCAPDataset(mcap_file, image_topic)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize voxel utility
    vox_util = utils.vox.VoxelUtil(
        Z=200,
        Y=200,
        X=200,
        scene_centroid=torch.tensor([0., 0., 0.]).to(device),
        bounds=torch.tensor([-5., 5., -5., 5., -5., 5.]).to(device),
        device=device
    )

    # Run inference
    with torch.no_grad():
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

            # Visualize or save results
            # Example: save segmentation output
            seg_output = seg_e[0].cpu().numpy()  # Take first batch item
            cv2.imwrite(f'bev_seg_{batch_idx}.png', (seg_output[0] * 255).astype(np.uint8))
            
            # Break after first frame for testing
            if batch_idx == 0:
                break

if __name__ == "__main__":
    run_inference()