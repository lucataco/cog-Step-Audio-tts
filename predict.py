import os
from cog import BasePredictor, Input, Path
import time
import torch
import torchaudio
import subprocess
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer

MODEL_CACHE = "checkpoints"
MODEL_TOKENIZER = "https://weights.replicate.delivery/default/stepfun-ai/Step-Audio-Tokenizer/model.tar"
MODEL_TTS = "https://weights.replicate.delivery/default/stepfun-ai/Step-Audio-TTS-3B/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE + "/Step-Audio-Tokenizer"):
            download_weights(MODEL_TOKENIZER, MODEL_CACHE + "/Step-Audio-Tokenizer")
        if not os.path.exists(MODEL_CACHE + "/Step-Audio-TTS-3B"):
            download_weights(MODEL_TTS, MODEL_CACHE + "/Step-Audio-TTS-3B")

        self.encoder = StepAudioTokenizer(f"{MODEL_CACHE}/Step-Audio-Tokenizer")
        self.tts_engine = StepAudioTTS(f"{MODEL_CACHE}/Step-Audio-TTS-3B", self.encoder)

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize into speech",
            default="（RAP I set out on the journey of freedom, chasing that distant dream, breaking free from the shackles of bondage, letting my soul drift with the wind, every step is full of power, every moment is extremely shining, the belief in freedom is burning, illuminating the direction of my progress!"
        ),
        speaker_name: str = Input(
            description="Speaker name", default="闫雨婷",
            choices=["闫雨婷", "闫雨婷RAP", "闫雨婷VOCAL"]
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        output_path = "/tmp/generated_speech.wav"

        # Generate speech
        output_audio, sr = self.tts_engine(text, speaker_name)
        
        # Save the audio file
        torchaudio.save(output_path, output_audio, sr)
        
        return Path(output_path) 