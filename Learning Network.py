import os
import re
import joblib
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["Normal", "HighTemp", "Moisture", "Damaged", "HighTemp_Damaged",
               "Moisture_Damaged", "HighTemp_Moisture", "All_Combined"]
DATA_FOLDER = "D:\\Charge_Data_Output"
MODEL_SAVE_PATH = "hybrid_classifier_strict.keras"
SCALER_SAVE_PATH = "timeseries_scaler.joblib"
IMAGE_SHAPE = (120, 120, 1)
TIME_SERIES_LENGTH = 500


def load_hybrid_data(data_folder, class_names):
    all_images, all_timeseries, all_labels = [], [], []
    class_to_id = {name: i for i, name in enumerate(class_names)}
    if not os.path.exists(data_folder):
        return np.array([]), np.array([]), np.array([])
    filenames = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
    for filename in filenames:
        match = re.match(r"([a-zA-Z_]+)_\d+\.mat", filename)
        if match and match.group(1) in class_to_id:
            label_id = class_to_id[match.group(1)]
            try:
                with h5py.File(os.path.join(data_folder, filename), 'r') as f:
                    if 'inverted_charge_map' in f and 'time_series_signal' in f:
                        img = f['inverted_charge_map'][()].T
                        ts_signal = f['time_series_signal'][()].flatten()
                        if len(ts_signal) != TIME_SERIES_LENGTH:
                            ts_signal = np.resize(ts_signal, TIME_SERIES_LENGTH)
                        all_images.append(img)
                        all_timeseries.append(ts_signal)
                        all_labels.append(label_id)
            except:
                continue
    return np.array(all_images)[..., np.newaxis], np.array(all_timeseries), np.array(all_labels)


def build_hybrid_model(img_shape, ts_length, num_classes):
    img_in = tf.keras.layers.Input(shape=img_shape, name="image_input")
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(img_in)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    cnn_out = tf.keras.layers.Flatten()(x)
    ts_in = tf.keras.layers.Input(shape=(ts_length,), name="timeseries_input")
    y = tf.keras.layers.Dense(108, activation='relu')(ts_in)
    y = tf.keras.layers.Dense(32, activation='relu')(y)
    fnn_out = tf.keras.layers.Dense(16, activation='relu')(y)
    combined = tf.keras.layers.Concatenate()([cnn_out, fnn_out])
    z = tf.keras.layers.Dense(128, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.3)(z)
    z = tf.keras.layers.Dense(64, activation='relu')(z)
    final_out = tf.keras.layers.Dense(num_classes, activation='softmax')(z)
    return tf.keras.Model(inputs=[img_in, ts_in], outputs=final_out)


if __name__ == '__main__':
    X_img, X_ts_raw, y_int = load_hybrid_data(DATA_FOLDER, CLASS_NAMES)
    if len(X_img) > 0:
        y_cat = tf.keras.utils.to_categorical(y_int, len(CLASS_NAMES))
        X_tr_img, X_te_img, X_tr_ts_raw, X_te_ts_raw, y_tr, y_te = train_test_split(
            X_img, X_ts_raw, y_cat, test_size=0.2, stratify=y_int
        )
        if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            scaler = joblib.load(SCALER_SAVE_PATH)
            X_te_ts = scaler.transform(X_te_ts_raw)
        else:
            scaler = StandardScaler()
            X_tr_ts = scaler.fit_transform(X_tr_ts_raw)
            X_te_ts = scaler.transform(X_te_ts_raw)
            joblib.dump(scaler, SCALER_SAVE_PATH)
            model = build_hybrid_model(IMAGE_SHAPE, TIME_SERIES_LENGTH, len(CLASS_NAMES))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit({"image_input": X_tr_img, "timeseries_input": X_tr_ts}, y_tr, epochs=100, batch_size=32,
                      verbose=0)
            model.save(MODEL_SAVE_PATH)

        preds = model.predict({"image_input": X_te_img, "timeseries_input": X_te_ts}, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_te, axis=1)

        print(f"{'True State':<20} | {'Predicted State':<20} | {'Confidence'}")
        print("-" * 65)
        for i in range(len(y_pred)):
            print(f"{CLASS_NAMES[y_true[i]]:<20} | {CLASS_NAMES[y_pred[i]]:<20} | {preds[i][y_pred[i]] * 100:.2f}%")