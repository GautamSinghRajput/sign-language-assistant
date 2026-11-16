# Sign Language Recognition and Translation Assistant

**Project ID:** CSEAIML2026-P26

A real-time system that watches a webcam, recognises American Sign Language (ASL) signs,
and shows the result as text and speech. It is built to make campus communication easier
for hearing- and speech-impaired students, faculty and staff.

---

## About the Project

Many hearing- and speech-impaired people find it hard to take part in classroom and
administrative conversations. This project uses computer vision to read signs from a normal
webcam and turn them into text and spoken audio. A reverse mode (typed or spoken text to sign
visuals) is planned.

**Primary users:** hearing- and speech-impaired students, faculty and staff.

---

## Features

- Real-time recognition from a normal webcam (no special hardware)
- Static model (MLP): fingerspelling letters A to Z from single frames
- Dynamic model (LSTM): moving signs from 30-frame sequences
- Works at different distances and hand positions (features are normalised)
- Smoothing (confidence limit and majority vote) to stop flickering
- On-screen text with word building, and speech output using pyttsx3 (works offline)
- Live latency display in milliseconds

---

## How It Works

```
Webcam frame
   -> MediaPipe Hands finds 21 points on each hand
   -> Points are normalised (wrist at origin, scaled by hand size)
   -> Single frame     -> MLP  -> static letter
   -> Last 30 frames   -> LSTM -> moving letter
   -> Confidence limit + majority vote
   -> Text on screen -> Speech
```

Using hand points instead of raw images makes the model faster and less affected by lighting
and background.

---

## Tech Stack

| Part | Tool |
|------|------|
| Language | Python 3.11 |
| Hand tracking | MediaPipe Hands |
| Video and images | OpenCV |
| Models | TensorFlow / Keras (MLP and LSTM) |
| Evaluation | scikit-learn, Matplotlib |
| Speech output | pyttsx3 |

---

## Project Structure

```
common.py          shared feature extraction and normalisation
extract.py         dataset (photos and videos) -> hand points (.npz)
train.py           trains and evaluates the static or dynamic model
recognize.py       live webcam recognition with text and speech
requirements.txt   Python libraries
data/              extracted data (not uploaded)
models/            trained models, labels, confusion matrices
```

---

## Getting Started

### Requirements

- A computer with a webcam
- Python 3.11 (newer versions may not work with MediaPipe or TensorFlow)

### Installation

```
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux use `python3.11 -m venv venv` and `source venv/bin/activate` instead.

### Dataset

The datasets are not included in this repository because of their size. Download the ASL
datasets from their original sources. Each letter needs its own folder:

```
SignAlphaSet/        static photos
    A/  img1.jpg ...
    B/  ...
dynamic_folder/      moving signs
    J/  clip1.avi ...
    Z/  ...
```

---

## Usage

```
python extract.py --static path/to/SignAlphaSet --dynamic path/to/dynamic_folder
python train.py --mode static
python train.py --mode dynamic
python recognize.py
```

### Keyboard controls (live app)

| Key | Action |
|-----|--------|
| M | Switch between Static and Dynamic mode |
| Space | Finish the current word |
| Backspace | Delete the last letter |
| S | Speak the sentence |
| C | Clear everything |
| Q | Quit |

---

## Evaluation

`train.py` prints accuracy, precision, recall and F1 score for each letter and saves a
confusion matrix in the `models/` folder.

| Model | Test accuracy | Live test (new user) | Average latency |
|-------|---------------|----------------------|-----------------|
| Static (A-Z) | to be added | to be added | to be added |
| Dynamic | to be added | to be added | to be added |

---

## Roadmap

- [x] Hand-point extraction from photos and videos
- [x] Static fingerspelling model (A-Z)
- [x] Live webcam recognition with text and speech output
- [ ] Dynamic (moving sign) model trained and evaluated
- [ ] Reverse mode: text or speech to sign visuals
- [ ] "Add a new sign" tool with retraining
- [ ] Streamlit interface
- [ ] Usability testing with accessibility-focused users
- [ ] Final evaluation report

---

## Limitations

- Trained on a limited dataset, so accuracy may drop for new users, backgrounds and lighting.
- Covers ASL letters for now. A bigger word vocabulary needs more data.
- Real sign language also uses facial expression and body movement, which this version does not use.

---

## References

Studied for ideas (the code in this repository was written independently):

- Handy: Sign Language Detection by Simon Dwyer (MIT) - https://github.com/sdwyersec/Handy-Sign-Language-Detection
- SignLens: Real-Time ASL Recognition by SiD-array (MIT) - https://github.com/SiD-array/SignLens-RealTime-ASL-Recognition
- MediaPipe Hands by Google - https://developers.google.com/mediapipe

---

## Team
| Gautam Singh Rajput 
| Abhay Chauhan 
| Kulbhan Singh 

