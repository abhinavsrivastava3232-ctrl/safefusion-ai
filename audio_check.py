"""
Audio Check - Standalone script to test microphone and YAMNet audio classification.
Lists default input device, opens microphone at 16 kHz mono, and every 0.5 seconds
for 20 seconds prints RMS level, top 5 YAMNet classes, and max distress score.
"""

import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import urllib.request
import csv
import io
import time

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_DURATION = 1.0  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
PROCESS_INTERVAL = 0.5  # seconds
TOTAL_DURATION = 20.0  # seconds

# Distress keywords to check
DISTRESS_KEYWORDS = {
    "scream", "shout", "yell", "crying",
    "glass", "alarm", "siren"
}


def list_default_device():
    """Print the default input device name."""
    try:
        device_info = sd.query_devices(kind='input')
        print(f"Default input device: {device_info['name']}")
        print(f"  Sample rate: {device_info['default_samplerate']} Hz")
        print(f"  Channels: {device_info['max_input_channels']}")
    except Exception as e:
        print(f"Error querying default device: {e}")


def load_yamnet_model():
    """Load YAMNet model from TensorFlow Hub."""
    try:
        print("Loading YAMNet model from TensorFlow Hub...")
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("YAMNet model loaded successfully")
        return yamnet_model
    except Exception as e:
        print(f"Error loading YAMNet model: {e}")
        return None


def load_class_names():
    """Load AudioSet class labels from public GitHub URL."""
    try:
        print("Loading AudioSet class labels...")
        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        with urllib.request.urlopen(url, timeout=10) as response:
            csv_text = response.read().decode("utf-8")
        reader = csv.reader(io.StringIO(csv_text))
        next(reader)  # skip header row
        class_names = [row[2] for row in reader]
        print(f"AudioSet labels loaded ({len(class_names)} classes)")
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None


def calculate_rms(audio_chunk):
    """Calculate RMS level of audio chunk."""
    return np.sqrt(np.mean(audio_chunk ** 2))


def get_top_predictions(scores, class_names, k=5):
    """Get top k predictions with their scores."""
    top_scores = tf.math.top_k(scores, k=k)
    top_indices = top_scores.indices.numpy()[0]
    top_values = top_scores.values.numpy()[0]
    
    predictions = []
    for idx, score in zip(top_indices, top_values):
        if idx < len(class_names):
            predictions.append((class_names[idx], float(score)))
    
    return predictions


def get_max_distress_score(scores, class_names, keywords):
    """Get maximum score among classes matching distress keywords."""
    max_score = 0.0
    matched_label = ""
    
    for idx, score in enumerate(scores.numpy()[0]):
        if idx < len(class_names):
            label = class_names[idx]
            label_lower = label.lower()
            for keyword in keywords:
                if keyword in label_lower:
                    if score > max_score:
                        max_score = float(score)
                        matched_label = label
    
    return max_score, matched_label


def main():
    """Main entry point for audio check."""
    print("=" * 60)
    print("Audio Check - Microphone and YAMNet Test")
    print("=" * 60)
    
    # List default device
    list_default_device()
    print()
    
    # Load YAMNet model
    yamnet_model = load_yamnet_model()
    if yamnet_model is None:
        print("Failed to load YAMNet model. Exiting.")
        return
    
    # Load class names
    class_names = load_class_names()
    if class_names is None:
        print("Failed to load class names. Exiting.")
        return
    
    print()
    
    # Initialize audio stream
    try:
        print(f"Opening audio stream: {SAMPLE_RATE} Hz, {CHANNELS} channel(s)")
        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE
        )
        stream.start()
        print("Audio stream started successfully")
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        return
    
    print()
    print(f"Running for {TOTAL_DURATION} seconds, printing every {PROCESS_INTERVAL} seconds...")
    print("=" * 60)
    
    # Audio buffer
    audio_buffer = []
    last_process_time = time.time()
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < TOTAL_DURATION:
            # Read audio chunk
            audio_chunk, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("Warning: Audio buffer overflow")
            
            audio_buffer.extend(audio_chunk.flatten())
            
            # Keep buffer at most 2 seconds
            if len(audio_buffer) > CHUNK_SIZE * 2:
                audio_buffer = audio_buffer[-(CHUNK_SIZE * 2):]
            
            # Process at interval
            current_time = time.time()
            if current_time - last_process_time >= PROCESS_INTERVAL:
                if len(audio_buffer) >= CHUNK_SIZE:
                    # Get last 1 second of audio
                    waveform = np.array(audio_buffer[-CHUNK_SIZE:], dtype=np.float32)
                    
                    # Calculate RMS
                    rms = calculate_rms(waveform)
                    
                    # Run YAMNet inference
                    scores, embeddings, spectrogram = yamnet_model(waveform)
                    
                    # Get top 5 predictions
                    top_predictions = get_top_predictions(scores, class_names, k=5)
                    
                    # Get max distress score
                    max_distress_score, matched_label = get_max_distress_score(
                        scores, class_names, DISTRESS_KEYWORDS
                    )
                    
                    # Print results
                    print(f"\n[{current_time - start_time:.1f}s] RMS: {rms:.6f}")
                    print("Top 5 YAMNet predictions:")
                    for i, (label, score) in enumerate(top_predictions, 1):
                        print(f"  {i}. {label}: {score:.4f}")
                    if max_distress_score > 0:
                        print(f"Max distress score: {max_distress_score:.4f} (matched: {matched_label})")
                    else:
                        print("Max distress score: 0.0000 (no match)")
                    print("-" * 60)
                
                last_process_time = current_time
            
            # Small sleep to prevent tight loop
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    
    finally:
        # Cleanup
        stream.stop()
        stream.close()
        print("\nAudio stream closed")
        print("Audio check complete")


if __name__ == "__main__":
    main()
