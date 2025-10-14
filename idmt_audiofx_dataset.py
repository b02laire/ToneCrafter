import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def parse_xml_dir(list_dir):
    """Parse all XMLs in a Lists/<Effect>/ folder and return mapping: key -> fileID"""
    mapping = {}
    for xml_path in Path(list_dir).glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for af in root.findall(".//audiofile"):
            key = (
                af.findtext("instrument"),
                af.findtext("instrumentsetting"),
                af.findtext("playstyle"),
                af.findtext("midinr"),
                af.findtext("polytype"),
            )
            fileid = af.findtext("fileID")
            mapping[key] = fileid
    return mapping


class IDMTAudioFXDataset(Dataset):
    """
    Pairs dry (NoFX) and wet (Effect) signals from the IDMT SMT-Audio-Effects dataset.
    """

    def __init__(self, lists_root, samples_root, effect="Chorus", window_size=2048):
        self.window_size = window_size
        self.effect = effect
        self.lists_root = Path(lists_root)
        self.samples_root = Path(samples_root)

        # Parse XML mappings for both
        dry_map = parse_xml_dir(self.lists_root / "NoFX")
        wet_map = parse_xml_dir(self.lists_root / effect)

        # Match shared keys (same instrument, note, playstyle)
        shared_keys = sorted(set(dry_map.keys()) & set(wet_map.keys()))

        # Build list of existing file pairs on disk
        pairs = []
        for key in shared_keys:
            dry_id = dry_map[key]
            wet_id = wet_map[key]
            dry_path = self.samples_root / "NoFX" / f"{dry_id}.wav"
            wet_path = self.samples_root / effect / f"{wet_id}.wav"
            if dry_path.exists() and wet_path.exists():
                pairs.append((dry_path, wet_path, key))

        self.pairs = pairs
        print(f"[IDMT FX] Found {len(pairs)} paired samples for effect: {effect}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dry_path, wet_path, key = self.pairs[idx]
        dry, _ = torchaudio.load(dry_path)
        wet, _ = torchaudio.load(wet_path)

        # # Convert to mono (average channels)
        dry, wet = dry.mean(0), wet.mean(0)

        # Normalize
        dry = dry / (dry.abs().max() + 1e-9)
        wet = wet / (wet.abs().max() + 1e-9)

        # Random crop or pad
        L = self.window_size
        if len(dry) > L:
            start = np.random.randint(0, len(dry) - L)
            dry = dry[start:start+L]
            wet = wet[start:start+L]
        else:
            pad = L - len(dry)
            dry = torch.nn.functional.pad(dry, (0, pad))
            wet = torch.nn.functional.pad(wet, (0, pad))

        return dry.float(), wet.float()
