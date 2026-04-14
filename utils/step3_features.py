import numpy as np
from scipy.ndimage import zoom

# -------------------------------
# Resize weather → match LULC
# -------------------------------
def resize_to_match(source, target_shape):
    zoom_factors = (
        target_shape[0] / source.shape[0],
        target_shape[1] / source.shape[1]
    )
    return zoom(source, zoom_factors)


if __name__ == "__main__":

    print("🔹 Loading cropped data...")

    lulc = np.load("../data/processed/lulc_crop.npy")
    fire = np.load("../data/labels/fire_crop.npy")
    weather = np.load("../data/weather/weather_small.npy")

    print("LULC:", lulc.shape)
    print("Fire:", fire.shape)
    print("Weather:", weather.shape)

    # -------------------------------
    # Resize weather
    # -------------------------------
    weather_resized = resize_to_match(weather, lulc.shape)

    print("Weather resized:", weather_resized.shape)

    # -------------------------------
    # Normalize
    # -------------------------------
    lulc = lulc / np.max(lulc)
    weather_resized = weather_resized / np.max(weather_resized)

    # -------------------------------
    # Stack features
    # -------------------------------
    features = np.stack([lulc, weather_resized], axis=0)

    # Add channel to label
    fire = np.expand_dims(fire, axis=0)

    print("Final feature shape:", features.shape)
    print("Final label shape:", fire.shape)

    # Save for training
    np.save("../data/processed/features.npy", features)
    np.save("../data/labels/labels.npy", fire)

    print("✅ Feature stack saved")