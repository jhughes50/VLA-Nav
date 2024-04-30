import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import TextImagePathDataset
from scripts.model import CLIPP


texts = ["This is a test", "This is another test", "This is the final test"]
image_paths = ["/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000000.png","/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000001.png",
               "/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000002.png"]

dataset = TextImagePathDataset(image_paths, texts)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CLIPP(512, 512, 512)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for text_embed, image_embed in data_loader:
        optimizer.zero_grad()
        logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images = model(text_embed, image_embed)
        # Compute loss
        loss = model.compute_loss(logits_per_text_path, logits_per_path_text, logits_per_images_path, logits_per_path_images)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
