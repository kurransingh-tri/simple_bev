import torch
from nets.segnet import Segnet
import utils.vox
import numpy as np

def export_segnet_to_onnx(checkpoint_path, onnx_path):
    Z, Y, X = 200, 8, 200
    scene_centroid_py = np.array([0.0, 1.0, 0.0]).reshape([1, 3])
    scene_centroid = torch.from_numpy(scene_centroid_py).float()
    bounds = (-50, 50, -5, 5, -50, 50)
    vox_util = utils.vox.Vox_util(Z=Z, Y=Y, X=X, scene_centroid=scene_centroid, bounds=bounds)

    # Initialize model
    model = Segnet(
        Z=200, Y=8, X=200,
        use_radar=False,
        use_lidar=False,
        rand_flip=False,
        latent_dim=128,
        vox_util=vox_util,
        encoder_type="res101"
    )


    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()
    # Create dummy inputs
    batch_size = 1
    num_cameras = 1
    dummy_input = {
        'rgb_camXs': torch.randn(1, 1, 3, 800, 1280).float(),
        'pix_T_cams': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1).float(),
        'cam0_T_camXs': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1).float(),
    }
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input['rgb_camXs'], 
         dummy_input['pix_T_cams'],
         dummy_input['cam0_T_camXs'],
         None), # rad_occ_mem0
        onnx_path,
        input_names=['rgb_camXs', 'pix_T_cams', 'cam0_T_camXs'],
        output_names=['raw_feat', 'feat', 'segmentation', 'instance_center', 'instance_offset'],
        opset_version=16,
        dynamic_axes={
            'rgb_camXs': {0: 'batch_size'},
            'pix_T_cams': {0: 'batch_size'},
            'cam0_T_camXs': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    checkpoint_path = "checkpoints/8x5_5e-4_rgb12_22:43:46/model-000025000.pth"
    onnx_path = "segnet_model.onnx"
    export_segnet_to_onnx(checkpoint_path, onnx_path)
    print(f"Model exported to {onnx_path}")