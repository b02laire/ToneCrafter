import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from idmt_audiofx_dataset import IDMTAudioFXDataset


def spectral_distance(x, y, n_fft=1024, hop_length=None, eps=1e-7):
    """
    Spectral distance loss between waveforms x and y.
    Based on: log(|STFT(x)|^2 + eps) - log(|STFT(y)|^2 + eps)

    Args:
        x, y: Tensors of shape (batch, time)
        n_fft: FFT size
        hop_length: hop length for STFT (defaults to n_fft // 4)
        eps: numerical stability constant
    Returns:
        scalar tensor (mean L1 loss)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    X = torch.stft(x.to("cuda"), n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to("cuda"),
                   return_complex=True)
    Y = torch.stft(y.to("cuda"), n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to("cuda"),
                   return_complex=True)

    # Magnitude squared (power)
    X_mag = X.abs().pow(2)
    Y_mag = Y.abs().pow(2)

    # Log compression
    X_log = torch.log(X_mag + eps)
    Y_log = torch.log(Y_mag + eps)

    loss = torch.nn.functional.l1_loss(X_log, Y_log)
    return loss


class GuitarMLP(nn.Module):
    def __init__(self, window_size=2048, hidden_size=1024, num_layers=6):
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
        return self.net(x)


# Dataset and loader
dataset = IDMTAudioFXDataset(
    lists_root="dataset/Gitarre polyphon/Lists",
    samples_root="dataset/Gitarre polyphon/Samples",
    window_size=2048
)
loader = DataLoader(dataset, batch_size=96, shuffle=True, num_workers=6, persistent_workers=True)

# Model + optimizer
model = GuitarMLP(window_size=2048).to("cuda")
model.load_state_dict(torch.load("models/test_cuda.pt"))
# model.compile()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = spectral_distance

# Training
print(time.strftime('%H:%M:%S', time.localtime()))
for epoch in range(4*60*60*5):
    start = time.time()
    total_loss = 0
    for dry, wet in loader:
        dry = dry.to("cuda")
        wet = wet.to("cuda")

        pred = model(dry)
        loss = criterion(pred, wet)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_time = time.time() - start

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{4*60*60*5}], Loss: {loss.item():.4f}, Time: {epoch_time:.4f}")

print(time.strftime('%H:%M:%S', time.localtime()))
torch.save(model.state_dict(), Path("models/test_cuda.pt"))

model.eval()

dry, sr = torchaudio.load("Clean.wav")
dry = dry.to("cuda")

# Split into windows and process
window_size = 2048
wet = []
for i in range(0, dry.size(-1), window_size):
    chunk = dry[:, i:i + window_size]
    if chunk.size(-1) < window_size:
        chunk = torch.nn.functional.pad(chunk, (0, window_size - chunk.size(-1)))
    with torch.no_grad():
        out = model(chunk)
    wet.append(out.cpu())

wet = torch.cat(wet, dim=-1)
torchaudio.save("chorus_mlp_output.wav", wet, sr)
