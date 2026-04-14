import numpy as np
from step1_load_data import load_lulc, load_fire_txt, load_weather

# Crop size (SAFE)
CROP_SIZE = 512

def crop_center(data, size):
    h, w = data.shape
    start_x = h // 2 - size // 2
    start_y = w // 2 - size // 2
    return data[start_x:start_x+size, start_y:start_y+size]


if __name__ == "__main__":

    print("🔹 Loading full data...")
    lulc = load_lulc("../data/processed/land.tif")
    fire = load_fire_txt("../data/labels/fire.txt", lulc.shape, lulc)
    weather = load_weather("../data/weather/weather1.nc")

    print("🔹 Cropping center region (512x512)...")

    lulc_crop = crop_center(lulc, CROP_SIZE)
    fire_crop = crop_center(fire, CROP_SIZE)

    print("LULC crop:", lulc_crop.shape)
    print("Fire crop:", fire_crop.shape)

    # Save (VERY IMPORTANT for next steps)
    np.save("../data/processed/lulc_crop.npy", lulc_crop)
    np.save("../data/labels/fire_crop.npy", fire_crop)
    np.save("../data/weather/weather_small.npy", weather)

    print("✅ Cropped data saved")