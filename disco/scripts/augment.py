import librosa
import random
import numpy as np


class Augmentations:
    """Audio augmentations"""

    def __init__(self):
        pass

    def time_stretch(self, audio, rate=None):
        if rate is None:
            rate = random.uniform(0.8, 1.2)  # random rate between 0.8x and 1.2x
        return librosa.effects.time_stretch(y=audio, rate=rate)

    def pitch_shift(self, audio, sr, n_steps=None):
        if n_steps is None:
            n_steps = random.uniform(-1, 1)  # random pitch shift between -1 and 1
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    def add_noise(self, audio, noise_level=None):
        if noise_level is None:
            noise_level = random.uniform(0.001, 0.005)  # add random noise
        noise = np.random.randn(len(audio))
        return audio + noise_level * noise

    def random_crop(self, audio, sr, target_duration):
        target_length = int(sr * target_duration)
        if len(audio) >= target_length:
            start = random.randint(0, len(audio) - target_length)
            return audio[start : start + target_length]
        return audio

    def change_volume(self, audio, volume_change=None):
        if volume_change is None:
            volume_change = random.uniform(0.5, 1.5)  # random volume change
        return audio * volume_change

    def pad_clip(self, audio, sr, target_duration):
        target_length = int(sr * target_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            audio = audio[:target_length]
        return audio

    def apply_augmentations(self, audio, sr, target_duration):
        audio = self.time_stretch(audio)
        audio = self.pitch_shift(audio, sr)
        audio = self.add_noise(audio)
        audio = self.change_volume(audio)
        audio = self.random_crop(audio, sr, target_duration)
        audio = self.pad_clip(audio, sr, target_duration)
        return audio
