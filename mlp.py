import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from idmt_audiofx_dataset import IDMTAudioFXDataset


class GuitarMLP(nn.Module):
    def __init__(self, window_size=2048, hidden_size=2048, num_layers=6):
        super().__init__()
        layers = []
        in_dim = window_size

        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size

        layers += [nn.Linear(in_dim, window_size)]  # output same size as input
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, window_size)
        return self.net(x)+x




# Dataset and loader
dataset = IDMTAudioFXDataset(
    lists_root="dataset/Gitarre polyphon/Lists",
    samples_root="dataset/Gitarre polyphon/Samples",
    window_size=2048
)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

# Model + optimizer
model = GuitarMLP(window_size=2048)
# model.load_state_dict(torch.load("models/test_id.pt"))
model.compile()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training
for epoch in range(300):
    total_loss = 0
    for dry, wet in loader:

        pred = model(dry)
        loss = criterion(pred, wet)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"{time.strftime("%H:%M:%S", time.localtime())}"
          f" Epoch {epoch+1}: Loss = {total_loss/len(loader):.6f}")
torch.save(model.state_dict(), Path("models/test_id.pt"))

model.eval()

dry, sr = torchaudio.load_with_torchcodec("Clean.wav")


# Split into windows and process
window_size = 2048
wet = []
for i in range(0, dry.size(-1), window_size):
    chunk = dry[:, i:i+window_size]
    if chunk.size(-1) < window_size:
        chunk = torch.nn.functional.pad(chunk, (0, window_size - chunk.size(-1)))
    with torch.no_grad():
        out = model(chunk)
    wet.append(out.cpu())

wet = torch.cat(wet, dim=-1)
torchaudio.save_with_torchcodec("chorus_mlp_output.wav", wet, sr)

