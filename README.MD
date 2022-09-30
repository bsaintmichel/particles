# Particles

A simple implementation of

* Particle detection using [Trackpy](http://soft-matter.github.io/trackpy/v0.5.0/tutorial.html)
* A (2d) pair correlation $g(\rho)$ using home-made scripts used [from a previous paper](https://doi.org/10.1039/D1SM01771C)

Brice Saint-Michel, Laboratoire Navier (`bsaintmichel`, still around by googlemail.com)

--------------------------------------------------------

### Input : Image files, either .png, .jpg but also bioformats (.lsm, ...)

* Standard images are loaded directly
* Bio-formats can be loaded using `functions.loadlsm()`

--------------------------------------------------------

### How does the programme work ?

Particle tracking is based on the Crocker & Grier (JCIS 1990) algorithm, kindly implemented in Python by the Trackpy community.

The pair correlation function $g(\rho)$ is computed by measuring, for each candidate particle, the distances to every other particle. I then make a 'histogram' of such distances and average these histograms for each candidate particle. The programe is taking into account the fact that particles can be close to the border of the image, in the same spirit as what Eric Weeks has done ([see here](http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html)) in 2d. I do not think my programme works at all in 3d. 

I have no idea if my code is efficient, I suppose you can't really beat a complexity of $N_{\rm parts}^2$, but I implemented it with `numba`, so it should be quicker than Python for large sets.

-----------------------------------------------------

### Dependencies 

* [Bokeh](https://bokeh.org/) for plotting
* [PIMS] () for loading scientific images
* [Trackpy](http://soft-matter.github.io/trackpy/) for particle detection
* [numba]() to make the 2d $g(\rho)$ code faster

Install through Pip : 
```
pip install <package-name>
```

Or through Anaconda :
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install <package-name>
```
--------------------------------------------------------