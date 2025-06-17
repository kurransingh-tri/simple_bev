import torch
from nets.segnet import Segnet
from utils.vox import VoxelUtil

def export_segnet_to_onnx(checkpoint_path, onnx_path):
    # Initialize model
    model = Segnet(
        Z=200, Y=200, X=200,
        use_radar=False,
        use_lidar=False,
        rand_flip=False,
        latent_dim=128,
        encoder_type="res101"
    )
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    num_cameras = 1
    dummy_input = {
        'rgb_camXs': torch.randn(batch_size, num_cameras, 3, 800, 1280),
        'pix_T_cams': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cameras, 1, 1),
        'cam0_T_camXs': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cameras, 1, 1),
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input['rgb_camXs'], 
         dummy_input['pix_T_cams'],
         dummy_input['cam0_T_camXs'],
         None,  # vox_util
         None), # rad_occ_mem0
        onnx_path,
        input_names=['rgb_camXs', 'pix_T_cams', 'cam0_T_camXs'],
        output_names=['raw_feat', 'feat', 'segmentation', 'instance_center', 'instance_offset'],
        opset_version=12,
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