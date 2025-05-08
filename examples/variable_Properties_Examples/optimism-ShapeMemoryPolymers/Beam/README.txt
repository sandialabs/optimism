********** SMP READ ME **********

This project utilizes the behaviors of shape memory polymers with the property optimization process. A random assignment of temperature dependent properties can be altered to affect a geometry's response to temperature and load cycling.

This code in particular takes in a geometry file, which does not have to be seeded (and will not consider the seeded values if there are any), stretches it at some high temperature, cools the geometry, releases the load, and then raises the temperature again to relax the geometry. The random seed of the properties (the density values between 0 and 1) are in a pseudo-random seed. This can be changed via the "key" variable at the bottom of the script. 
