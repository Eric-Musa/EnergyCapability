import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / 'data'

solar_path = DATA_DIR / 'solar' / 'pv_open_2020.csv'


def effective_capacity(capacities, distance_to_transmission_km, loss_per_km=0.00005):
    return capacities * (1 - (distance_to_transmission_km * loss_per_km))


def regional_capacity(coordinate, longitudes, latitudes, capacities, radius=1):
    dists = cdist([coordinate], np.vstack([longitudes, latitudes]).T).flatten()
    cap_in_radius = capacities[dists < radius]
    return cap_in_radius.sum()


solar_df = pd.read_csv(solar_path)

lon = solar_df['longitude'].values
lat = solar_df['latitude'].values
cap = solar_df['capacity_mw'].values * solar_df['capacity_factor'].values
area = solar_df['area_sq_km'].values
dist = solar_df['distance_to_transmission_km'].values
eff_cap = effective_capacity(cap, dist)


def solar_alpha(capacities, factor=0.01, min_a=0):
    return (capacities - capacities.min()) / (capacities.max() - capacities.min()) * factor + min_a


def solar_size(area_sq_km, factor=10, min_s=0):
    return ((area_sq_km - area_sq_km.min()) / (area_sq_km.max() - area_sq_km.min()) * factor + min_s) ** 2


solar_afactor = 0.25
solar_sfactor = 1

coordinate = [-77.4, 39.5]


plt.scatter(lon, lat, s=1)
plt.scatter(*coordinate, s=100)
plt.show()
# plt.scatter(lon, lat, alpha=solar_alpha(eff_cap, solar_afactor), s=solar_size(area, solar_sfactor))
# ax1.scatter(lon, lat, alpha=solar_alpha(eff_cap, solar_afactor), s=solar_size(eff_cap, solar_sfactor))
# ax2.scatter(lon, lat, alpha=solar_alpha(area, solar_afactor), s=solar_size(area, solar_sfactor))


# from sklearn.neural_network import MLPRegressor
# from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.gaussian_process import GaussianProcessRegressor

# x = np.vstack([lon, lat]).T
# y = np.vstack([eff_cap, area]).T

# fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))

# layers = [10, 10]
# # solar_mlp = MLPRegressor(layers, max_iter=5000, tol=0.000001, verbose=True)
# # solar_mlp.fit(x, y)
# knn = KNeighborsRegressor(50, leaf_size=100)
# knn.fit(x, y)

# # mlp_y_pred = solar_mlp.predict(x)
# mlp_y_pred = knn.predict(x)
# pred_eff_cap = mlp_y_pred[:, 0]
# pred_area = mlp_y_pred[:, 1]

# ax1.scatter(lon, lat, alpha=solar_alpha(area, solar_afactor), s=solar_size(area, solar_sfactor))
# ax2.scatter(lon, lat, alpha=solar_alpha(pred_eff_cap, solar_afactor), s=solar_size(pred_area, solar_sfactor))
# fig.show()
# # solar_gpr = GaussianProcessRegressor()
# # solar_gpr.fit(x, y)


