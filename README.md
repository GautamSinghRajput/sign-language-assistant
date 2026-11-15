# Sign Language Recognition and Translation Assistant

Project ID: CSEAIML2026-P26

A computer-vision system that reads American Sign Language (ASL) from a webcam
and shows the result as text and speech, to make campus communication easier
for hearing- and speech-impaired people.

## Features
- Real-time recognition from a webcam using MediaPipe hand landmarks
- **Static model (MLP):** fingerspelling letters A-Z from single frames
- **Dynamic model (LSTM):** moving signs from 30-frame video sequences
- Text output on screen and speech output with `pyttsx3`
- Smoothing (confidence limit + majority vote) to stop flickering
- Planned: reverse mode (text/speech to sign), "add a new sign" tool, Streamlit interface

## Setup (Windows, Python 3.11)
```
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage
```
python extract.py --static path\to\SignAlphaSet --dynamic path\to\dynamic_folder
python train.py --mode static
python train.py --mode dynamic
python recognize.py
```
Keys in the live app: `M` switch mode, `Space` finish word, `Backspace` delete,
`S` speak, `C` clear, `Q` quit.

## Dataset
The ASL datasets are **not included** in this repository because of their size.
Download them from their original sources and point `extract.py` at the folders.

## Evaluation
`train.py` prints accuracy, precision, recall and F1 score, and saves a
confusion matrix in `models/`. The live app shows prediction latency in milliseconds.

## References
Studied for ideas (code was written independently):
- Handy: Sign Language Detection (MIT) by Simon Dwyer
- SignLens: Real-Time ASL Recognition (MIT) by SiD-array

## Author
YOUR NAME, YOUR COLLEGE (edit this line)
