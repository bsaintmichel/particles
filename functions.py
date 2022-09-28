# functions.py

import pandas as pd
import numpy as np
import pims
from numba import njit

def normalise_image(img, force_max=None):
    """
    Function that normalises an image
    """    
    if force_max is None:
        max_image = np.max(img)
    else:
        max_image = force_max
    
    img = np.double(img)
    img_temp = np.round(255*(img - np.min(img))/(max_image - np.min(img)))
    img_temp[img_temp > 255] = 255
    return img_temp

def loadLSM(path_to_file):
    """
    A bit of code to read a .lsm (confocal stack) file
    directly. 

    Args 
    ----
    > 'path_to_file' : the complete path to the .lsm file

    Returns 
    -------
    > 'frames' : the frames in question (N_x times N_y (times N_z) times N_t)
    > 'info' : the metadata of the picture  

    """
    frames   = pims.Bioformats(path_to_file)
    meta     = frames.metadata
    width    = meta.PixelsSizeX(0)
    height   = meta.PixelsSizeY(0)
    nframes  = meta.PixelsSizeT(0)
    pixsizeX = meta.PixelsPhysicalSizeX(0)
    pixsizeY = meta.PixelsPhysicalSizeY(0)
    dt       = meta.PixelsTimeIncrement(0)
    objective = meta.ObjectiveCorrection(0,0)
    magnification = meta.ObjectiveNominalMagnification(0,0)
    date = meta.ImageAcquisitionDate(0)
    nameraw  = path_to_file.split('/')
    filenameraw = nameraw[-1].split('.')
    filename = '.'.join(filenameraw[:-1])
    folder = '/'.join(nameraw[:-1])
    info = {'date': date, 'width': width, 'height': height, 'nframes': nframes, 'pixsize': pixsizeX, 'pixsize2': None, 'objective': objective, 'magnification': magnification, 'dt': dt, 'filename':filename, 'folder':folder}
    print('>> ' + str(nframes) + ' frames of size : ' + str(height) + ' px (height) x ' + str(width) + ' px (width), delta t is : ' + str(dt) + ' s')
    if (pixsizeX != pixsizeY):
        print('>> Quick reminder: X pixel size (' + str(pixsizeX) + ' µm) and Y pixel size (' + str(pixsizeY)  + ' µm) are different')
        info['pixsize2'] = pixsizeY
    else:
        print('>> Pixel size is: ' + str(pixsizeX) + ' µm')
    return frames, info

############################################################################
### Data processing functions ("main" functions) --------------------------------------------------------------
############################################################################

def particle_pair(particles, props, particle_pixel_size, rhos=200):
    """
    A wrapper that calls 'particle_pair_numba' to compute 
    the particle pair function of a confocal image.
    
    Args
    ----
    > 'particles' : the result of trackpy.locate (a Pandas DataFrame)
    > 'props' : the metadata of your image stack. It should be a dictionary containing at least the 
    image height and width.
    > 'particle_pixel_size' : float, in pixels ; self_explanatory
    > 'rhos' : the distances at which you want to evaluate the pair correlation function. 
    Can be an int, or an array, in which case I will treat your rhos as the 'edges' of the
    bins at which the g(rho) is computed

    Returns 
    ----
    > 'rho_ctrs' : the center of the 'bins' at which the g(rho) function is evaluated
    > 'g' : the pair correlation function

    """
    
    particles_num = np.zeros((len(particles), 2))
    particles_num[:,0], particles_num[:,1] = particles['x'], particles['y']
    props_num = np.array([props['height'], props['width']])

    if type(rhos) is int:
        rho_edges = np.linspace(0.01,10*particle_pixel_size,rhos+1)
    else:
        rho_edges = rhos

    # Calling the numba particle pair function
    # Then normalise the x output
    rho_ctrs, g = particle_pair_numba(particles_num, props_num, rho_edges)
    rho_ctrs = rho_ctrs/particle_pixel_size

    return rho_ctrs, g

@njit
def particle_pair_numba(particles_num, props_num, rho_edges):
    """
    My function computing particle pairs. Note : it does take into account the 
    borders of the image in the normalisation (so no worries about particles
    being close to the image edge).

    Args 
    ----
    > 'particles_num' : Nx2 list/ndarray  of particle coordinates
    > 'props_num' : list of 2 values : [height, width]
    > 'particle_pixel_size' : size of a particle in pixels (theoretical or experimental)
    > 'rho_edges' : edges of the 'bins' at which the g(rho) function is evaluates 

    Returns
    -------
    > 'rho_ctrs' [np.1Darray] : the center of the 'bins' at which the g(rho) function is evaluated
    > 'g' [np.1Darray] : the pair correlation function
    
    """

    # Parsing the inputs 
    rho_ctrs = (rho_edges[:-1] + rho_edges[1:])/2
    xp, yp = particles_num[:,0], particles_num[:,1]
    nparticles = xp.size
    height, width = props_num[0], props_num[1]

    # Well ... so the technique consists in doing repeat histograms for the neighbors of each particle
    particle_density = nparticles/(height*width)
    norm_factor = normalization(particles_num, rho_edges, props_num)*particle_density
    g = np.zeros(rho_ctrs.size)
    
    for pno in range(nparticles):
        dx, dy = xp - xp[pno], yp - yp[pno]
        dists = np.sqrt(dx**2 + dy**2)
        hist, bins = np.histogram(dists, bins=rho_edges)
        g += hist/norm_factor[pno,:]
    
    g = g/nparticles
    return rho_ctrs, g
    
    
@njit
def normalization(particles_num, rho_edges, props_num):
    """ Normalization of the g(rhos) function due to the particle edges
     
    When the "rho" (interparticle distance) at which the g(rho) function is evaluated
    intersects the boundaries of the frame, this function
    computes the adequate geometry normalization so that g(rho) tends to one at long distances
    
    Args
    ----
    > particle_coords : Nx2 numpy array
    > rho_edges : 1xM numpy array
    > props_num : list of [height, width] of the image

    Returns
    -------
    > normalisation : NxM array of normalisation factors (for each particle, and each rho)

    # NOTE : normalization no longer works if you can intersect more than two segments
    # (i.e. when rho exceeds half of the smallest ROI dimension) so be careful !
    """

    # Parsing the inputs 
    xp, yp = particles_num[:,0], particles_num[:,1]
    rho_ctrs = 0.5*(rho_edges[:-1] + rho_edges[1:])
    nparticles, nrho = xp.size, rho_ctrs.size
    height, width = props_num[0], props_num[1]

    # Separating the two places where we know we don't have particles
    # 1/ the bubble (d'uh), # 2/ outside of the picture (d'uh)
    a_corner = np.zeros(nparticles)
    result = np.zeros((nparticles, nrho))
    yb, xb = np.minimum(yp, height - yp), np.minimum(xp, width  - xp)
    
    for rh_no in range(nrho): 
        xbn = np.minimum(xb/rho_ctrs[rh_no], np.ones(np.shape(xb)))
        ybn = np.minimum(yb/rho_ctrs[rh_no], np.ones(np.shape(yb)))

        # Are we hitting one edge/two edges/the bubble ?
        two_edges = xbn**2 + ybn**2 < 1
        one_edge = np.logical_and(np.logical_or(xbn**2 < 1, ybn**2 < 1), np.logical_not(two_edges))
        
        # Computing the relevant areas when edges are intersected
        a_corner[two_edges] = np.pi/2 + np.arccos(ybn[two_edges]) + np.arccos(xbn[two_edges])
        a_corner[one_edge] = 2*(np.arccos(xbn[one_edge]) + np.arccos(ybn[one_edge]))
        
        result[:,rh_no] = (2*np.pi - a_corner)*(rho_edges[rh_no+1]**2 - rho_edges[rh_no]**2)/2

    return result

def percus_yevick(rhos, phi):
    """
    Percus Yevick (analytical) pair correlation
    for hard spheres. Taken from Wikipedia (C code)
    
    Args 
    ----
    > rhos : normalised distances between particles (divided by particle diameter) [1D np.array]
    > phi : your volume fraction [FLOAT]

    Returns
    ----
    > g_PY : percus Yevick volume fraction [1D np.array, same size as rhos]

    """
    qr = np.pi*rhos
    a = (1+2*phi)**2/(1-phi)**4
    b = -6*phi*(1+phi/2)**2/(1-phi)**4
    c = phi/2*(1+2*phi)**2/(1-phi)**4
    A = 2*qr
    Q1 = a/A**2*(np.sin(A) - A*np.cos(A))
    Q2 = b/A**3*(2*A*np.sin(A) + (2-A**2)*np.cos(A) - 2)
    Q3 = c/A**5*(-A**4*np.cos(A) + 4*((3*A**2-6)*np.cos(A) + A*(A**2-6)*np.sin(A)+6))
    g = Q1 + Q2 + Q3  

    return 1/(1+24*phi*g/A)



def export_g_rho(rho_ctrs, g_exp, g_PY, file_name='data.csv'):
    """
    Function that exports g(r) both experimental and Percus Yevick.
    
    """
    data = pd.DataFrame({'rho':rho_ctrs, 'g_exp':g_exp, 'g_PY': g_PY})
    data.to_csv(file_name)