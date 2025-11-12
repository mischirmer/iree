"""Read a 1000-float classification output file, convert logits to
probabilities and display the top-k ImageNet predictions.

Usage:
  python analyze_output.py [classification.bin] [--top K]

If TensorFlow is installed the script will use
`tf.keras.applications.resnet50.decode_predictions` to produce human
readable labels. Otherwise it falls back to the ImageNet JSON mapping.
"""

import argparse
import json
import urllib.request
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def load_imagenet_mapping():
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    )
    return json.load(urllib.request.urlopen(url))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", default="classification.bin", help="float32 output file")
    p.add_argument("--top", type=int, default=3, help="top-K predictions to show")
    args = p.parse_args()

    try:
        arr = np.fromfile(args.file, dtype=np.float32)
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        return

    if arr.size != 1000 and arr.size != 1 * 1000:
        print(f"Unexpected number of elements in '{args.file}': {arr.size} (expected 1000)")
        # attempt to continue if reshape possible
    logits = arr.reshape(-1)[:1000]
    probs = softmax(logits.astype(np.float32))

    # Try to use TensorFlow's decode_predictions if available
    try:
        from tensorflow.keras.applications.resnet50 import decode_predictions

        # decode_predictions expects a batch dimension
        batch = probs.reshape(1, -1)
        decoded = decode_predictions(batch, top=args.top)[0]
        print("Top predictions (from tf.keras decode_predictions):")
        for cls, name, prob in decoded:
            print(f"{cls}\t{name}\tprob={prob:.6f}")
        return
    except Exception:
        # Fall back to JSON mapping if TF not available
        class_idx = load_imagenet_mapping()
        topk = probs.argsort()[::-1][: args.top]
        print("Top predictions (fallback mapping):")
        for i in topk:
            label = class_idx[str(int(i))][1]
            print(f"{int(i)}\t{label}\tprob={probs[int(i)]:.6f}\tlogit={logits[int(i)]:.2f}")


if __name__ == "__main__":
    main()