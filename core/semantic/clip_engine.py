import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

TEXT_CLASSES = [
    "a woman is being attacked",
    "a violent situation",
    "a person holding a weapon",
    "a dangerous environment",
    "a normal safe scene"
]

def get_clip_threat(frame):
    image = Image.fromarray(frame)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize(TEXT_CLASSES).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image_input, text_input)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # If any unsafe label has higher probability → threat
    unsafe_score = probs[0] + probs[1] + probs[2]
    safe_score = probs[3]

    return float(unsafe_score) * 100
