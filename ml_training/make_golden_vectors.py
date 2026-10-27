"""
Generates deterministic TEST FIXTURES (not training data, not performance numbers) used to prove that
the Java and Python feature extractors agree. Writes app/src/test/resources/:
  golden_signals.csv   case,t,red,green,blue
  golden_features.csv  case,feature_index,feature_name,value
Re-run after any change to the feature algorithm, then run the Java tests.
"""
import os
import numpy as np
from ppg_features import extract_features, FEATURE_NAMES

OUT = os.path.join(os.path.dirname(__file__), "..", "app", "src", "test", "resources")


def synth(seed, hr_hz, resp_hz, seconds, fs, jitter, noise, hrv):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs + rng.uniform(-jitter, jitter, n) * (1 / fs)
    t = np.sort(t)
    t = t - t[0] + 0.0
    ph = 2 * np.pi * hr_hz * t + hrv * np.sin(2 * np.pi * 0.1 * t)
    pulse = np.sin(ph) + 0.35 * np.sin(2 * ph + 0.6)
    resp = np.sin(2 * np.pi * resp_hz * t)
    red = 232 + 1.4 * pulse + 0.9 * resp + noise * rng.standard_normal(n) + 0.2 * t / seconds
    green = 118 + 1.0 * pulse + 0.6 * resp + noise * rng.standard_normal(n)
    blue = 41 + 0.25 * pulse + 0.15 * resp + 0.5 * noise * rng.standard_normal(n)
    return t, red, green, blue


CASES = {
    "regular_72bpm": synth(1, 1.20, 0.25, 30, 30.0, 0.0, 0.05, 0.0),
    "slow_55bpm_hrv": synth(2, 0.92, 0.20, 30, 28.5, 0.15, 0.10, 0.6),
    "jittery_fast_105bpm": synth(3, 1.75, 0.33, 32, 24.0, 0.35, 0.15, 0.2),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "golden_signals.csv"), "w") as fs_, \
            open(os.path.join(OUT, "golden_features.csv"), "w") as ff:
        fs_.write("case,t,red,green,blue\n")
        ff.write("case,feature_index,feature_name,value\n")
        for name, (t, r, g, b) in CASES.items():
            for row in zip(t, r, g, b):
                fs_.write(name + "," + ",".join(repr(float(v)) for v in row) + "\n")
            feats = extract_features(t, r, g, b)
            for i, (n, v) in enumerate(zip(FEATURE_NAMES, feats)):
                ff.write("%s,%d,%s,%s\n" % (name, i, n, repr(float(v))))
            print(name, "g_dom_freq_hz=%.2f hr_from_peaks=%.1f" % (feats[0], feats[11]))


if __name__ == "__main__":
    main()
