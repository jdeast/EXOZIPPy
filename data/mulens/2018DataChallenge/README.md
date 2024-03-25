Light curves from the 2018 WFIRST (now Roman) Data Challenge:

# About the Data Challenge

http://www.microlensing-source.org/data-challenge/


# Change log

17 August 2018: The event_info.txt file contained incorrect extinction and extinction dispersion values in both bands. The file has been updated and is now correct. Additionally, information about the color-surface brightness relation, which can be used to estimate source angular diameters, has been added to this readme below.


# Lightcurves

Lightcurves are zipped in lc.tar.gz - this will unzip to the directory lc/

Columns: BJD Magnitude Error



# Ephemeris files

There is one ephemeris file for each filter: wfirst_ephemeris_W149.txt and wfirst_ephemeris_Z087.txt. They store the position of WFIRST in the solar system at the times of the observations in each filter. Positions are in units of AU

Columns: BJD X_EQ Y_EQ Z_EQ X_ECL Y_ECL Z_ECL

X,Y,Z_EQ are cartesian coordinates in the J2000 equatorial frame with the solar system barycenter at (0,0,0)

X,Y,Z_ECL are cartesian coordinates in the ICRS ecliptic plane frame with the solar system barycenter at (0,0,0)

Both frames are right handed, the X axis is common to both and increases from the barycenter towards the vernal equinox, Z points towards the north celestial or ecliptic pole, respectively.




# Event Information

There are a few pieces of information that may need to be known for each event that are not in the lightcurve files. These are stored in event_info.txt

Columns: Event_name Event_number RA_(deg) Dec_(deg) Distance A_W149 sigma_A_W149 A_Z087 sigma_A_Z087


Distance, A_W149/Z087 are an estimate of the distance and extinction in each band of the red clump stars. sigma_A_W149/Z087 are dispersions in the extinction.


# Estimating angular diameters of sources

The angular diameter of the source can be estimated from the color-surface brightness relation for dereddened Z087-W149 color:

$\log \theta_{W149=0} = 0.986 + 0.366 (Z087-W149)_0$

where \theta_{W149=0} is the angular diameter at zero apparent W149 magnitude (measured in milli-arcsec) and (Z087-W149)_0 is the dereddened color. The angular diameter is related to the angular diameter at zero magnitude by

$\log \theta_{W149=0} = \log \theta + 0.2 W149_0$

where $\theta$ is the angular *diameter* (in mas) and $W149_0$ is the dereddened W149 magnitude. See [Adams, Boyajian & von Braun (2018)](https://ui.adsabs.harvard.edu//#abs/2018MNRAS.473.3608A/abstract) for an overview of such relations, and [Yoo et al. 2004](https://ui.adsabs.harvard.edu//#abs/2004ApJ...603..139Y/abstract) for an example of how they are used to estimate Einstein radii. The above relation is only valid for $-0.3<=(Z087-W149)_0<0.6$, and can be assumed to have an intrinsic scatter about the relation of 0.034 in $\log \theta_{W149=0}$.



# Problems?

If you spot any problems with the data set, please open an issue on this github repository, or contact Matthew Penny: penny1@lsu.edu.


