import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
import os

def pack_bits_to_uint32_array(binary_array):
    pad_len = (32 - (len(binary_array) % 32)) % 32
    padded = np.pad(binary_array, (0, pad_len), mode='constant')
    chunks = padded.reshape(-1, 32)
    uint32_arr = np.zeros(chunks.shape[0], dtype=np.uint32)
    for i in range(32):
        uint32_arr |= (chunks[:, i].astype(np.uint32) << i)
    return uint32_arr

def pack_ternary_to_uint32_array(ternary_array):
    mapped = ternary_array & 0x03
    pad_len = (16 - (len(mapped) % 16)) % 16
    padded = np.pad(mapped, (0, pad_len), mode='constant')
    chunks = padded.reshape(-1, 16)
    uint32_arr = np.zeros(chunks.shape[0], dtype=np.uint32)
    for i in range(16):
        uint32_arr |= (chunks[:, i].astype(np.uint32) << (i * 2))
    return uint32_arr

def write_uint32_array(f, name, array):
    f.write(f"const uint32_t {name}[{len(array)}] = {{\n    ")
    elems = [f"0x{v:08X}" for v in array]
    for i in range(0, len(elems), 8):
        chunk = ", ".join(elems[i:i+8])
        if i + 8 < len(elems):
            f.write(chunk + ",\n    ")
        else:
            f.write(chunk + "\n")
    f.write("};\n\n")

def write_float_array(f, name, array):
    f.write(f"const float {name}[{len(array)}] = {{\n    ")
    elems = [f"{v:.5f}f" for v in array]
    for i in range(0, len(elems), 8):
        chunk = ", ".join(elems[i:i+8])
        if i + 8 < len(elems):
            f.write(chunk + ",\n    ")
        else:
            f.write(chunk + "\n")
    f.write("};\n\n")

def run_hdc_pipeline(dataset_name, X_train_raw, X_test_raw, Y_train, Y_test, header_path):
    print(f"\n{'='*50}\nProcessing Dataset: {dataset_name}\n{'='*50}")
    
    n_samples, n_features = X_train_raw.shape
    C_classes = len(np.unique(Y_train))
    
    print(f"  -> Training samples : {n_samples}")
    print(f"  -> Test samples     : {X_test_raw.shape[0]}")
    print(f"  -> Raw features (d) : {n_features}")
    print(f"  -> Classes (C)      : {C_classes}")

    # ============================================================
    # Phase 2 — HDC Encoding (QuantHD logic)
    # ============================================================
    D = 10000          # Hypervector dimensionality
    n_levels = 10      # Number of quantization levels
    np.random.seed(42)

    ID_hvs = np.random.randint(0, 2, size=(n_features, D), dtype=np.uint8)
    min_val = np.min(X_train_raw)
    max_val = np.max(X_train_raw)
    intervals = np.linspace(min_val, max_val, n_levels + 1)

    L_hvs = np.zeros((n_levels, D), dtype=np.uint8)
    L_hvs[0] = np.random.randint(0, 2, size=D, dtype=np.uint8)
    flip_count = D // n_levels
    for i in range(1, n_levels):
        flip_indices = np.random.choice(D, size=flip_count, replace=False)
        L_hvs[i] = np.copy(L_hvs[i-1])
        L_hvs[i][flip_indices] = 1 - L_hvs[i][flip_indices]

    def encode_dataset(X):
        X_levels = np.digitize(X, intervals) - 1
        X_levels = np.clip(X_levels, 0, n_levels - 1)
        encoded = np.zeros((X.shape[0], D), dtype=np.uint8)
        for i in range(X.shape[0]):
            bound = np.bitwise_xor(ID_hvs, L_hvs[X_levels[i]])
            bundled_sum = np.sum(bound, axis=0)
            encoded[i] = np.where(bundled_sum > (n_features / 2), 1, 0)
        return encoded

    print(f"Encoding training and test data...")
    X_train_enc = encode_dataset(X_train_raw)
    X_test_enc = encode_dataset(X_test_raw)

    # ============================================================
    # Phase 3 — QuantHD Training (Ternary Model)
    # ============================================================
    def quantize_ternary(class_float_matrix):
        sigma = np.std(class_float_matrix)
        boundary = 0.42 * sigma
        ternary_hvs = np.zeros_like(class_float_matrix, dtype=np.int8)
        ternary_hvs[class_float_matrix > boundary] = 1
        ternary_hvs[class_float_matrix < -boundary] = -1
        return ternary_hvs

    def predict_ternary(X_enc, C_hvs_ternary):
        X_bip = np.where(X_enc == 1, 1, -1)
        scores = np.dot(X_bip, C_hvs_ternary.T)
        return np.argmax(scores, axis=1)

    Class_hvs_float = np.zeros((C_classes, D), dtype=np.float32)
    alpha = 0.05
    for c in range(C_classes):
        class_mask = (Y_train == c)
        X_bipolar = np.where(X_train_enc[class_mask] == 1, 1, -1)
        Class_hvs_float[c] = np.sum(X_bipolar, axis=0) * alpha

    Class_hvs_ternary = quantize_ternary(Class_hvs_float)
    init_acc = accuracy_score(Y_train, predict_ternary(X_train_enc, Class_hvs_ternary))
    print(f"  -> Initial Train Acc: {init_acc * 100:.2f}%")

    epochs = 10
    print(f"Retraining for {epochs} epochs...")
    for epoch in range(epochs):
        preds = predict_ternary(X_train_enc, Class_hvs_ternary)
        errors = 0
        for i in range(n_samples):
            y_true = Y_train[i]
            y_pred = preds[i]
            if y_true != y_pred:
                x_bip = np.where(X_train_enc[i] == 1, 1, -1)
                Class_hvs_float[y_true] += (alpha * x_bip)
                Class_hvs_float[y_pred] -= (alpha * x_bip)
                errors += 1
        Class_hvs_ternary = quantize_ternary(Class_hvs_float)

    test_preds = predict_ternary(X_test_enc, Class_hvs_ternary)
    test_acc = accuracy_score(Y_test, test_preds)
    print(f"  -> Final Test Acc : {test_acc * 100:.2f}%")

    # ============================================================
    # Phase 4 — Hardware-Aware Exporter
    # ============================================================
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write(f"#ifndef HDC_MODEL_{dataset_name.upper()}_H\n")
        f.write(f"#define HDC_MODEL_{dataset_name.upper()}_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write(f"#define HDC_D {D}\n")
        f.write(f"#define HDC_FEATURES {n_features}\n")
        f.write(f"#define HDC_LEVELS {n_levels}\n")
        f.write(f"#define HDC_CLASSES {C_classes}\n")
        
        uint32_per_hv = (D + 31) // 32
        f.write(f"#define HDC_UINT32_PER_HV {uint32_per_hv}\n")
        ternary_uint32_per_hv = (D + 15) // 16
        f.write(f"#define HDC_TERNARY_UINT32_PER_HV {ternary_uint32_per_hv}\n\n")
        
        write_float_array(f, "HDC_intervals", intervals)
        
        id_hvs_packed = np.array([pack_bits_to_uint32_array(hv) for hv in ID_hvs]).flatten()
        write_uint32_array(f, "HDC_ID_hvs", id_hvs_packed)
        
        l_hvs_packed = np.array([pack_bits_to_uint32_array(hv) for hv in L_hvs]).flatten()
        write_uint32_array(f, "HDC_L_hvs", l_hvs_packed)
        
        class_hvs_packed = np.array([pack_ternary_to_uint32_array(hv) for hv in Class_hvs_ternary]).flatten()
        write_uint32_array(f, "HDC_Class_hvs", class_hvs_packed)
        
        n_test_samples = min(20, X_test_raw.shape[0])
        f.write(f"#define HDC_TEST_SAMPLES {n_test_samples}\n\n")
        
        test_X_flat = X_test_raw[:n_test_samples].flatten()
        write_float_array(f, "HDC_test_X", test_X_flat)
        
        test_Y_flat = Y_test[:n_test_samples]
        f.write(f"const uint32_t HDC_test_Y[{n_test_samples}] = {{")
        f.write(", ".join(map(str, test_Y_flat)))
        f.write("};\n\n")
        
        f.write(f"#endif // HDC_MODEL_{dataset_name.upper()}_H\n")
    print(f"Exported to '{header_path}'")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1. UCI-HAR Dataset
    train_df = pd.read_csv("dataset/train.csv")
    test_df  = pd.read_csv("dataset/test.csv")
    drop_cols = ['Activity', 'subject']
    X_train_har = train_df.drop(columns=drop_cols).values.astype(np.float32)
    X_test_har  = test_df.drop(columns=drop_cols).values.astype(np.float32)
    le = LabelEncoder()
    Y_train_har = le.fit_transform(train_df['Activity'].values)
    Y_test_har  = le.transform(test_df['Activity'].values)
    
    run_hdc_pipeline("har", X_train_har, X_test_har, Y_train_har, Y_test_har, "dataset/HDC_model_har.h")

    # 2. Digits Dataset
    digits = load_digits()
    X_dig = digits.data.astype(np.float32)
    Y_dig = digits.target
    X_train_dig, X_test_dig, Y_train_dig, Y_test_dig = train_test_split(X_dig, Y_dig, test_size=0.2, random_state=42)
    run_hdc_pipeline("digits", X_train_dig, X_test_dig, Y_train_dig, Y_test_dig, "dataset/HDC_model_digits.h")

    # 3. Breast Cancer Dataset
    cancer = load_breast_cancer()
    X_can = cancer.data.astype(np.float32)
    Y_can = cancer.target
    X_train_can, X_test_can, Y_train_can, Y_test_can = train_test_split(X_can, Y_can, test_size=0.2, random_state=42)
    run_hdc_pipeline("cancer", X_train_can, X_test_can, Y_train_can, Y_test_can, "dataset/HDC_model_cancer.h")

