"""
Use case for the architecture of the Galactic Model Penalty.
"""
import exozippy as mmexo

my_star = mmexo.Star(
    mass=1.0,  # M_Sun
    distance=6.0,  # kpc
    gal_coords=[0.0, -0.5],  # (ell, b) deg
    mu_hel_gal=[-6, 0.],  # mas/yr in Galactic coordinate system.
    extinction={'I': 1.5},  # A_I in mag
    age=8.0  # Gyr
)

print('Galactic Model chi^2 Penalty: ', my_star.get_galactic_model_penalty())
print('Probability of being a DISK star: ', my_star.disk_probability)
print('Probability of being a BULGE star: ', my_star.bulge_probability)

# Print a dictionary with the penalties from different parts of the models.
print(my_star.galactic_model_penalty_components)
