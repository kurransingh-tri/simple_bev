import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

class MCAPDataset(Dataset):
    """Dataset class for loading images from MCAP files"""
    def __init__(self, mcap_file, image_topic, transform=None):
        self.mcap_file = mcap_file
        self.image_topic = image_topic
        self.transform = transform
        self.frames = self._load_frame_indices()

    def _load_frame_indices(self):
        """Pre-scan MCAP file to get frame indices"""
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
        if len(image.shape) == 2:  # grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply any transforms
        if self.transform:
            image = self.transform(image)
            
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'timestamp': frame['log_time']
        }

class SimpleBEVFormer(nn.Module):
    """Simple BEVFormer implementation"""
    def __init__(self, img_size=(800, 1280), bev_size=(100, 100)):
        super().__init__()
        
        # Image encoder (ResNet-like)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # BEV projection head
        self.bev_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # Single channel BEV output
        )

    def forward(self, x):
        features = self.img_encoder(x)
        bev = self.bev_head(features)
        return bev

def train():
    # Configuration
    mcap_file = "cam0_ole.mcap"
    image_topic = "/cam0/image_compressed"
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    dataset = MCAPDataset(mcap_file, image_topic)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SimpleBEVFormer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Simple loss for demonstration

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            bev_output = model(images)
            
            # For demonstration, using a dummy target
            # In practice, you would need ground truth BEV labels
            dummy_target = torch.zeros_like(bev_output)
            
            # Compute loss
            loss = criterion(bev_output, dummy_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'bevformer_model.pth')

if __name__ == "__main__":
    train()