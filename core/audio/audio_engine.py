import threading
import time
import numpy as np

# Try importing TensorFlow libraries for Audio Classification
# If installation fails, the system falls back to MOCK_MODE for demonstration
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import pyaudio
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow or PyAudio not found. Running in MOCK MODE.")

class AudioSafetyEngine:
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode or (not TF_AVAILABLE)
        self.running = False
        self.current_audio_label = "Silence"
        self.current_audio_conf = 0.0
        
        # Threat classes YAMNet can detect (indices based on AudioSet)
        # 'Screaming': 0, 'Shouting': 1, 'Glass': 2, etc. (Simplification for this demo)
        self.threat_keywords = ['Screaming', 'Shout', 'Yell', 'Crying, sobbing', 'Glass']
        
        if not self.mock_mode:
            self.init_yamnet()
            self.init_mic()

    def init_yamnet(self):
        print("Loading YAMNet Audio Model... (This may take a moment)")
        # Load YAMNet from TensorFlow Hub (Standard for research)
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_map_path = self.model.class_map_path().numpy()
        self.class_names = self.class_names_from_csv(self.class_map_path)
        print("Audio Model Loaded.")

    def class_names_from_csv(self, class_map_csv_text):
        """Returns list of class names corresponding to score vector."""
        import csv
        import io
        class_names = []
        with io.TextIOWrapper(io.BytesIO(class_map_csv_text), encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    def init_mic(self):
        self.p = pyaudio.PyAudio()
        # YAMNet expects 16kHz mono audio
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  frames_per_buffer=1024)

    def start(self):
        self.running = True
        # Run audio listening in a separate thread so video doesn't freeze
        self.thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.thread.start()

    def audio_loop(self):
        """Continuous loop to listen and classify sounds."""
        print("Audio Engine Listening...")
        while self.running:
            if self.mock_mode:
                # SIMULATION FOR TESTING: Randomly trigger a scream every 10 seconds
                time.sleep(1) 
                if int(time.time()) % 20 == 0:
                    self.current_audio_label = "Screaming"
                    self.current_audio_conf = 0.95
                else:
                    self.current_audio_label = "Background Noise"
                    self.current_audio_conf = 0.0
                continue

            # REAL LOGIC
            try:
                # Read 0.975 seconds of audio (YAMNet window)
                waveform_data = self.stream.read(15600, exception_on_overflow=False)
                waveform = np.frombuffer(waveform_data, dtype=np.int16) / 32768.0
                
                # Predict
                scores, embeddings, spectrogram = self.model(waveform)
                prediction = np.mean(scores, axis=0)
                top_class_index = np.argmax(prediction)
                current_label = self.class_names[top_class_index]
                
                # Update shared variables
                self.current_audio_label = current_label
                self.current_audio_conf = prediction[top_class_index]
                
            except Exception as e:
                print(f"Audio Error: {e}")
                time.sleep(0.1)

    def get_status(self):
        """Returns tuple (Label, Is_Threat_Boolean)"""
        is_threat = any(keyword in self.current_audio_label for keyword in self.threat_keywords)
        return self.current_audio_label, is_threat, self.current_audio_conf

    def stop(self):
        self.running = False
        if not self.mock_mode:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()