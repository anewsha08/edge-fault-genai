"""Train fault classifiers on the feature dataset and report honest test accuracy.

Usage:
  python ml/train.py                      # split by motor load: train 0,1 HP / val 2 HP / test 3 HP
  python ml/train.py --split random       # random window split (leaky; shown to demonstrate inflation)

Outputs go to ml/out/: metrics.json, scaler.json, confusion matrices, and model.keras (if TensorFlow is installed).
"""
import argparse
import json
import os

import numpy as np

CLASSES_FALLBACK = ["normal", "inner", "ball", "outer"]


def split(d, mode, seed=0):
    y = d["y"]
    n = len(y)
    if mode == "load":
        load = d["load"]
        return np.where(load <= 1)[0], np.where(load == 2)[0], np.where(load == 3)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    return perm[:a], perm[a:b], perm[b:]


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_cm_plot(cm, classes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/features.npz")
    ap.add_argument("--out", default="ml/out")
    ap.add_argument("--split", choices=["load", "random"], default="load")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data, allow_pickle=False)
    X, y = d["X"].astype(np.float32), d["y"].astype(int)
    classes = [str(c) for c in d["classes"]] if "classes" in d else CLASSES_FALLBACK
    k = len(classes)
    tr, va, te = split(d, args.split, args.seed)
    print(f"split={args.split}: train={len(tr)} val={len(va)} test={len(te)}")

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    with open(os.path.join(args.out, "scaler.json"), "w") as f:
        json.dump({"mean": mu.tolist(), "std": sd.tolist(),
                   "feature_names": [str(s) for s in d["feature_names"]]}, f, indent=2)

    metrics = {"split": args.split, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}

    # Baseline: random forest (reference only; too big for the MCU)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=args.seed, class_weight="balanced", n_jobs=-1)
    rf.fit(Xs[tr], y[tr])
    rf_pred = rf.predict(Xs[te])
    metrics["random_forest_test_acc"] = float((rf_pred == y[te]).mean())
    cm_rf = confusion(y[te], rf_pred, k)
    save_cm_plot(cm_rf, classes, f"RandomForest ({args.split} split)", os.path.join(args.out, "cm_rf.png"))
    print(f"RandomForest test acc: {metrics['random_forest_test_acc']:.3f}")

    # Small MLP in Keras: this is the model that gets quantized and runs on the Cortex-M
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed: skipping MLP. (In Colab it is preinstalled.)")
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return

    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(k, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    counts = np.bincount(y[tr], minlength=k)
    cw_ = {i: float(len(tr) / (k * max(c, 1))) for i, c in enumerate(counts)}
    model.fit(Xs[tr], y[tr], validation_data=(Xs[va], y[va]), epochs=args.epochs, batch_size=64,
              class_weight=cw_, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    pred = model.predict(Xs[te], verbose=0).argmax(1)
    metrics["mlp_float_test_acc"] = float((pred == y[te]).mean())
    metrics["mlp_params"] = int(model.count_params())
    cm = confusion(y[te], pred, k)
    save_cm_plot(cm, classes, f"MLP float ({args.split} split)", os.path.join(args.out, "cm_mlp_float.png"))
    model.save(os.path.join(args.out, "model.keras"))
    np.savez(os.path.join(args.out, "split_idx.npz"), train=tr, val=va, test=te)
    print(f"MLP float test acc: {metrics['mlp_float_test_acc']:.3f}  ({metrics['mlp_params']} params)")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
