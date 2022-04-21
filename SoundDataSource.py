from scipy.io import wavfile


class SoundDataSource:
    def __init__(self, filepath):
        self.filepath = filepath
        self.samplerate, self.data = wavfile.read(filepath)






