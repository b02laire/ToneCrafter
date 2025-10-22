import math
import time

import torch
import torch.nn as nn
from torch.nn import MaxPool1d
from torch.utils.data import DataLoader
from idmt_audiofx_dataset import IDMTAudioFXClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GuitarFXClassifier(nn.Module):
    def __init__(self):
        super(GuitarFXClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, padding=1, kernel_size=3),
            nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(16, 32, padding=1, kernel_size=3),
            # nn.MaxPool1d(2),
            nn.Conv1d(32, 64, padding =1, kernel_size=3),
            # MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64* 2048, 12),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)


dataset = IDMTAudioFXClassification(
    samples_root="dataset/Gitarre polyphon/Samples",
    window_size=2048,
)

loader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True
)

model = GuitarFXClassifier().to(device)
# compile_mode = "max-autotune"
# compile_mode += "-no-cudagraphs" if torch.cuda.is_available() else ""
# model.compile(mode = "max-autotune-no-cudagraphs")
# model.compile()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    start = time.time()
    for audio, label in loader:
        audio = audio.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        pred = model(audio)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

    epoch_time = time.time() - start

    # if (epoch + 1) % 10 == 0 or epoch == 0:
    print(
        f"Epoch [{epoch + 1}/{4 * 60 * 60 * 5}], Loss: {loss.item():.4f},"
        f" Time: {epoch_time:.4f}"
    )
