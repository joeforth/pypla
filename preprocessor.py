# -*- coding: utf-8 -*-
import pims
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_isodata, threshold_li, threshold_otsu, threshold_minimum, threshold_local, threshold_yen, threshold_triangle, threshold_mean, rank, gaussian
from skimage.morphology import label, binary_closing, binary_dilation, binary_erosion, local_maxima
from skimage.measure import regionprops
from skimage.feature import peak_local_max
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, square, cube, watershed
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier


def folder_finder(dir_in):
    folders = os.listdir(dir_in)
    # Find folder names and remove system files
    folders_trim = []
    for i in range(0, len(folders)):
        if folders[i][0] != '.' and folders[i] != 'Icon\r':
            folders_trim.append(folders[i])
    return folders_trim
    

def data_org(tifdir_in):
    # List of files in tif directory
    files =  os.listdir(tifdir_in)

    # Find image names and remove system files
    files_trim = []
    for i in range(0, len(files)):
        if files[i][0] != '.' and files[i] != 'Icon\r':
            files_trim.append(files[i])
    files = files_trim

    # Get filenames and image resolutions
    images = pd.read_csv('./resolution.txt', sep='\t')
    filenames, dx, dy, dz = images.iloc[:,0].values, images.iloc[:,1].values, images.iloc[:,2].values, images.iloc[:,3]
    metadata = pd.DataFrame({'filename': filenames, 'dx':dx, 'dy':dy, 'dz':dz})

    # File names in resolution.txt currently have a space after them. Add an if loop that runs through
    # each file name and chops off a space if it's there
    files_trim = []
    for i in range(0, len(filenames)):
        if filenames[i][len(filenames[i])-1] == ' ':
            files_trim.append(filenames[i][0:len(filenames[i])-1])
        else:
            files_trim.append(filenames[i])
    metadata['filename'] = files_trim
    return files, metadata


def stackloader(filename, dir_in='', plot=True):
    # Load up a stack - assumes that PLA signal on channel 1, DAPI on channel 2
    if dir_in == '':
        data = pims.TiffStack('./' + filename)
    if dir_in != '':
        data = pims.TiffStack(dir_in + filename)
    # This may need removing to run on the Mac / non-Conda Python
    # data = data[0]

    nuclei = np.array(data[1::2])
    nps = np.array(data[::2])

    z_size, y_size, x_size = np.shape(nps)

    nrows = np.int(np.ceil(np.sqrt(z_size)))
    ncols = np.int(z_size // nrows + 1)

    # print z_size, nrows, ncols

    if plot == True:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

        for n in range(z_size):
            i = n // ncols
            j = n % ncols
            axes[i, j].imshow(nuclei[n], interpolation='nearest', cmap='gray')
            
        for ax in axes.ravel():
            if not(len(ax.images)):
                fig.delaxes(ax)
        fig.tight_layout()

        plt.show()
        plt.close()

    return nuclei, nps, nrows, ncols


def stack_equalise(stack_in, high_in):
    z_size, y_size, x_size = np.shape(stack_in)
    stack_max = stack_in.max(axis=0)
    stack_out = np.zeros_like(stack_in)
#     low, high = np.percentile(stack_max, (0, high_in))
    high = stack_max.max()
    for i in range(0, z_size):
        stack_out[i] = exposure.rescale_intensity(stack_in[i], in_range=(0,high))
    return stack_out
        

def threshold(image_in, thresh_type):
    if thresh_type == 'yen':
        print('Using Yen threshold')
        return threshold_yen(image_in)
    if thresh_type == 'triangle':
        print('Using Triangle threshold')
        return threshold_triangle(image_in)
    if thresh_type == 'minimum':
        print('Using Minimum threshold')
        return threshold_minimum(image_in)
    if thresh_type == 'otsu':
        print('Using Otsu threshold')
        return threshold_otsu(image_in)
    if thresh_type == 'li':
        print('Using Li threshold')
        return threshold_li(image_in)
    if thresh_type == 'isodata':
        print('Using isodata threshold')
        return threshold_isodata(image_in)
    if thresh_type == 'mean':
        return threshold_mean(image_in)
    if thresh_type == 'none':
        return image_in


def smooth(image_in, smooth_type, smooth_size, contrast = False):
    # smoothed = rank.median(np.copy(image_in), disk(smooth_size))
    if smooth_type == 'mean':
        print('running a mean filter')
        return rank.mean(np.copy(image_in), disk(smooth_size))
    if smooth_type == 'median':
        print('running a median filter')
        return rank.median(np.copy(image_in), disk(smooth_size))
    if smooth_type == 'gaussian':
        print('running a Gaussian filter')
        return gaussian(image_in, smooth_size)
    if smooth_type == 'none':   
        print('not running a filter')
        return np.copy(image_in)


def bandpass(image_in, blur_size):
    print('running high bandpass filter')
    im_blur = gaussian_filter(image_in, blur_size)
    im_out = np.zeros_like(image_in)
    # Now need to implement the subtraction correctly
    im_out = image_in - im_blur
    im_out[np.where(image_in < im_blur)] = 0
    return im_out


def contrast(image_in, cont_size):
    if cont_size > 0:
        print('enhancing contrast, width', str(cont_size))
        cont = rank.enhance_contrast(np.copy(image_in), disk(cont_size))
        return cont
    if cont_size == 0:
        return image_in


def contrast_nuclei(image_in, cont_size):
    if cont_size > 0:
        print('Enhancing contrast')
        return rank.enhance_contrast(np.copy(image_in), disk(cont_size))
    if cont_size == 0:
        print('Not enhancing contrast')
        return image_in


def clusterer(particles_in, max_radius):
    # Apply hierarchical clustering algorithm to connect particles identified through the stack 
    # Create data array in which clustered particles are averaged store computed values
    values_clustered = []
    if len(particles_in) == 0:
        particles_in['particle'] = []
    if len(particles_in) == 1:
        particles_in['particle'] = 1
    elif len(particles_in) > 1:
        # Clustering to eliminate maxima that are too close together
        positions = particles_in[['x', 'y', 'z']].values
        # Distance matrix is n-particles x n-particles in size - reckon it gives the interparticle separation
        # This gives the upper triangle of the distance matrix 
        dist_mat = dist.pdist(positions)
        link_mat = hier.linkage(dist_mat)
        # fcluster assigns each of the particles in positions a cluster to which it belongs
        cluster_idx = hier.fcluster(link_mat, max_radius, criterion='distance')
        particles_in['particle'] = cluster_idx
    return particles_in
        

def averager(particles_in, area_cutoff_lo, area_cutoff_hi):
    # Calculate weighted average position of particles
    values_clustered = []
    particles = np.unique(particles_in['particle'].values)
    for i in particles:
        current = particles_in[particles_in['particle'] == i]
        # Normalisation constant of weighted average
        norm = np.sum(current['intensity'])
        x_av = np.mean(current['x'])
        y_av = np.mean(current['y'])
        # z-value is the intensity-weighted value
        z_av = np.sum(current['intensity'] * current['z'])/norm
        a_total = np.sum(current['area'])
        # Geometric cut-off to particle size
        if a_total > area_cutoff_lo and a_total < area_cutoff_hi:
            values_clustered.append([x_av, y_av, z_av, a_total, norm, i])

    columns_clustered = ('x', 'y', 'z', 'a_total', 'intensity', 'particle')
    particles_clustered = pd.DataFrame(values_clustered, columns=columns_clustered)
    print('Found', len(particles_clustered), 'particles')
    return particles_clustered




