"""Shared helpers: turn MediaPipe hand points into numbers the models can learn from."""
import numpy as np
import mediapipe as mp

SEQ_LEN = 30            # frames per moving sign
STATIC_FEATURES = 63    # 21 hand points x (x, y, z)
DYN_FEATURES = 126      # two hands x 63

# Set True only if your dataset videos were recorded mirrored (like a selfie view).
MIRROR_INPUT = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def make_hands(static=False):
    return mp_hands.Hands(
        static_image_mode=static,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def get_hands(results):
    """List of hands (each 21x3 array), sorted left-to-right in the image."""
    if not results.multi_hand_landmarks:
        return []
    hands = [np.array([[p.x, p.y, p.z] for p in h.landmark], dtype=np.float32)
             for h in results.multi_hand_landmarks]
    return sorted(hands, key=lambda p: p[0, 0])


# ---------- STATIC letters (one photo = one hand shape) ----------
def static_features(hand):
    """Shift so the wrist is (0,0,0) and scale by hand size.
    Works for any hand position or distance from the camera."""
    centred = hand - hand[0]
    scale = np.linalg.norm(centred[9]) + 1e-6       # wrist to middle-finger base
    return (centred / scale).flatten().astype(np.float32)


# ---------- DYNAMIC letters (a short video = a movement) ----------
def raw_frame_features(results):
    """126 raw numbers for one frame (zeros if a hand is missing)."""
    out = np.zeros(DYN_FEATURES, dtype=np.float32)
    for i, h in enumerate(get_hands(results)[:2]):
        out[i * 63:(i + 1) * 63] = h.flatten()
    return out


def resample(seq, length=SEQ_LEN):
    """Pick `length` evenly spaced frames from a list of frames."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((length, DYN_FEATURES), dtype=np.float32)
    idx = np.linspace(0, len(seq) - 1, length).round().astype(int)
    return seq[idx]


def normalize_sequence(seq):
    """Normalise a whole sequence using the FIRST visible hand.
    The same shift and scale are used for every frame, so the path the
    hand travels (needed for letters like J and Z) is kept."""
    slots = np.array(seq, dtype=np.float32).reshape(len(seq), 2, 21, 3)
    present = np.abs(slots).sum(axis=(2, 3)) > 0            # (T, 2)
    ts, ss = np.nonzero(present)
    if len(ts) == 0:
        return slots.reshape(len(seq), DYN_FEATURES)
    hand = slots[ts[0], ss[0]]
    centre = hand[0].copy()
    scale = np.linalg.norm(hand[9] - hand[0]) + 1e-6
    slots = np.where(present[:, :, None, None], (slots - centre) / scale, 0.0)
    return slots.reshape(len(seq), DYN_FEATURES).astype(np.float32)
