# Models: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, distil-large-v3.5, large-v3-turbo, turbo
from faster_whisper import WhisperModel
import time

class Transcriber:
    def __init__(self, model_size="medium.en"):
        self.model_size = model_size
        # Run on GPU with FP16
        # self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # or run on GPU with INT8
        # self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribe_audio(self, audio_file):
        segments, info = self.model.transcribe(audio_file, beam_size=5)
        text = ""
        for segment in segments:
            text+=segment.text
        text = text.replace("\n", "")
        return text, info