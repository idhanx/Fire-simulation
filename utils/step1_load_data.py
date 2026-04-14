import rasterio
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load LULC raster (light)
# -------------------------------
def load_lulc(path):
    with rasterio.open(path) as src:
        data = src.read(1)  # read only 1 band
    return data


# -------------------------------
# 2. Convert fire.txt → raster mask
# -------------------------------
def load_fire_txt(path, shape, lulc):
    # Temporary: simulate fire from vegetation (high-risk areas)
    
    mask = np.zeros(shape, dtype=np.uint8)

    # Assume lower LULC values = vegetation (adjust if needed)
    mask[lulc < 50] = 1  

    return mask


# -------------------------------
# 3. Load weather (NetCDF - SAFE)
# -------------------------------
def load_weather(path):
    ds = xr.open_dataset(path)

    print("Available variables:", list(ds.data_vars))

    var_name = list(ds.data_vars)[0]

    # 🔥 IMPORTANT: Load only first time slice (low RAM)
    data = ds[var_name].isel(valid_time=0).values

    return data


# -------------------------------
# 4. Downsample for visualization (RAM safe)
# -------------------------------
def downsample(data, factor=10):
    return data[::factor, ::factor]


# -------------------------------
# 5. Visualization
# -------------------------------
def visualize(data, title):
    plt.imshow(data, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()


# -------------------------------
# MAIN TEST
# -------------------------------
if __name__ == "__main__":

    print("🔹 Loading LULC...")
    lulc = load_lulc("../data/processed/land.tif")
    print("LULC shape:", lulc.shape)

    print("\n🔹 Loading fire data...")
    fire = load_fire_txt("../data/labels/fire.txt", lulc.shape, lulc)
    print("Fire pixel count:", np.sum(fire))

    print("\n🔹 Loading weather data...")
    weather = load_weather("../data/weather/weather1.nc")
    print("Weather shape:", weather.shape)

    print("\n🔹 Showing visualizations (downsampled)...")
    visualize(downsample(lulc), "LULC (downsampled)")
    visualize(downsample(fire), "Fire Mask (downsampled)")