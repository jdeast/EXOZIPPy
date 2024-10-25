import os.path
import matplotlib.pyplot as plt

from plot_results import PlanetFitInfo

failure = PlanetFitInfo(os.path.join('temp_output', 'WFIRST.008.log'))
failure.plot_initial_pspl_fit()
plt.savefig('LC.008.initial_pspl_fit.png', dpi=300)

bad_planet = PlanetFitInfo(os.path.join('temp_output', 'WFIRST.193.log'))
bad_planet.plot_revised_pspl_fit()
plt.savefig('LC.193.initial_pspl_fit.png', dpi=300)

file = os.path.join('temp_output', 'WFIRST.004.log')
planet = PlanetFitInfo(file)

planet.plot_initial_pspl_fit()
plt.savefig('LC.004.initial_pspl_fit.png', dpi=300)
planet.plot_revised_pspl_fit()
plt.savefig('LC.004.revised_pspl_fit.png', dpi=300)
planet.mag_methods[1] = 'VBBL'
planet.plot_initial_planet_model()
plt.savefig('LC.004.initial_2L1S_guess.png', dpi=300)

#plt.show()
