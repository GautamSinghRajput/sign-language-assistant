"""Step 3: live webcam recognition.

Keys:  M = switch Static/Dynamic   SPACE = finish word   BACKSPACE = delete letter
       S = speak sentence          C = clear             Q = quit
Run:   python recognize.py
"""
import json
import os
import time
from collections import Counter, deque
import cv2
import numpy as np
from tensorflow import keras
from common import (make_hands, get_hands, static_features, raw_frame_features,
                    normalize_sequence, mp_hands, mp_draw, SEQ_LEN, MIRROR_INPUT)

CONF = {"static": 0.85, "dynamic": 0.85}   # how sure the model must be
NEED = {"static": 7, "dynamic": 5}         # votes needed out of the last VOTE_WINDOW
VOTE_WINDOW = 10
PREDICT_EVERY = 3                          # run the model every 3 frames


def load(mode):
    path = f"models/{mode}_model.keras"
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path), json.load(open(f"models/{mode}_labels.json"))


models = {m: load(m) for m in ("static", "dynamic")}
models = {m: v for m, v in models.items() if v}
if not models:
    raise SystemExit("No trained model found. Run train.py first.")
mode = "static" if "static" in models else "dynamic"

try:
    import pyttsx3
    tts = pyttsx3.init()
except Exception:
    tts = None

buffer = deque(maxlen=SEQ_LEN)
votes = deque(maxlen=VOTE_WINDOW)
sentence, word = [], ""
locked, latency_ms, frame_no = None, 0.0, 0
cap = cv2.VideoCapture(0)

with make_hands(static=False) as hands:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR_INPUT:
            frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for h in (res.multi_hand_landmarks or []):
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
        found = get_hands(res)
        buffer.append(raw_frame_features(res))
        frame_no += 1

        if not found:                      # hand lowered: reset
            votes.clear()
            locked = None
        elif frame_no % PREDICT_EVERY == 0:
            model, labels = models[mode]
            t0 = time.perf_counter()
            if mode == "static":
                probs = model.predict(static_features(found[0])[None], verbose=0)[0]
            elif len(buffer) == SEQ_LEN:
                seq = normalize_sequence(np.array(buffer))
                probs = model.predict(seq[None], verbose=0)[0]
            else:
                probs = None
            if probs is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                best = int(probs.argmax())
                votes.append(labels[best] if probs[best] >= CONF[mode] else None)
                top, count = Counter(votes).most_common(1)[0]
                if top is not None and count >= NEED[mode] and top != locked:
                    if mode == "static":
                        word += top
                    else:
                        sentence.append(top)
                        buffer.clear()
                    locked = top
                    votes.clear()

        disp = frame if MIRROR_INPUT else cv2.flip(frame, 1)   # always show a mirror view
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 75), (0, 0, 0), -1)
        shown = " ".join(sentence[-6:] + ([word + "_"] if word else []))
        cv2.putText(disp, shown or "Show a sign...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(disp, f"Mode: {mode.upper()}  Latency: {latency_ms:.0f} ms  "
                          "[M] mode [Space] word [S] speak [C] clear [Q] quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("Sign Language Assistant", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence, word = [], ""
        elif key == 32 and word:                       # space
            sentence.append(word)
            word = ""
        elif key in (8, 127):                          # backspace
            word = word[:-1]
        elif key == ord("m") and len(models) > 1:
            mode = "dynamic" if mode == "static" else "static"
            votes.clear(); buffer.clear(); locked = None
        elif key == ord("s") and tts:
            tts.say(" ".join(sentence + ([word] if word else [])))
            tts.runAndWait()

cap.release()
cv2.destroyAllWindows()
