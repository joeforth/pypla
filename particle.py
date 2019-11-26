# -*- coding: utf-8 -*-

import pims
import numpy as np
import os
import trackpy as tp
import matplotlib.pyplot as plt
from skimage.morphology import label, binary_closing, binary_dilation, binary_erosion, local_maxima
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, square, cube, watershed
import pandas as pd
from pypla import preprocessor
from pypla import vis
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier

from mpl_toolkits.axes_grid1 import make_axes_locatable


def nps_finder_tp_cluster(image_file, z_extent, filter_kernel, min_sep):
    files, metadata = preprocessor.data_org(tifdir)

    columns = ('x', 'y', 'z', 'a_total', 'intensity', 'particle', 'com_dist', 'trial')
    all_parts = pd.DataFrame([], columns=columns)

    for i in range(0,len(files)):
        print(i)
        print(files[i])
        frames = pims.TiffStack('./tifs/' + folders[m] + '/tifs/' + files[i])
        # nps = np.array(frames[::2])
        # nuclei = np.array(frames[1::2])
        features = tp.batch(nps[:], diameter=filter_kernel, separation=min_sep)

        # Option to add a mass cut
        # features = features[features['raw_mass'] > 0]

        if len(features) == 0:
            features['particle'] = []
        if len(features) == 1:
            features['particle'] = 1
        elif len(features) > 1:
            # Clustering to eliminate maxima that are too close together
            positions = features[['x', 'y', 'frame']].values
            # Distance matrix is n-particles x n-particles in size - reckon it gives the interparticle separation
            # This gives the upper triangle of the distance matrix 
            dist_mat = dist.pdist(positions)
            link_mat = hier.linkage(dist_mat)
            # fcluster assigns each of the particles in positions a cluster to which it belongs
            cluster_idx = hier.fcluster(link_mat, 5, criterion='distance')
            features['particle'] = cluster_idx

        n_parts = np.unique(features['particle'].values)
        values_clustered = []
        for j in n_parts:
            current = features[features['particle'] == j]
            if len(current) > z_extent:
                norm = np.sum(current['raw_mass'])
                x_av = np.mean(current['x'])
                y_av = np.mean(current['y'])
                # z-value is the intensity-weighted value
                z_av = np.sum(current['raw_mass'] * current['frame'])/norm
                a_total = np.sum(current['size'])
                    # Geometric cut-off to particle size
                if a_total > 1:
                    values_clustered.append([x_av, y_av, z_av, a_total, norm, j])

        columns_clustered = ('x', 'y', 'z', 'a_total', 'intensity', 'particle')
        particles_clustered = pd.DataFrame(values_clustered, columns=columns_clustered)
#         particles_clustered = particles_clustered[particles_clustered['intensity'] < upper_mass_cut]


def nps_finder_2d(stack_in, name_in, thresh_type, smooth_type, smooth_size=1, gauss_size=0, cont_size=0, sep_method='watershed', mass_cutoff=200, area_cutoff=0, max_radius=5, min_radius=1, test=True, plot=True, save=True):
    # np.copy() used because peak_local_max does weird things to the image histogram if we use the actual image
    print('mass_cutoff is', mass_cutoff)
    z_size, y_size, x_size = np.shape(stack_in)

    # Generate flatted image from stack
    max_int_proj = stack_in.max(axis=0)
    # Image to store labels
    labeled_image = np.zeros_like(max_int_proj)

    # Smooth data
    smoothed = preprocessor.smooth(np.copy(max_int_proj), smooth_type, smooth_size)
    smoothed = preprocessor.bandpass(np.copy(smoothed), gauss_size) 
    smoothed = preprocessor.contrast(np.copy(smoothed), cont_size)
    # Calculate threshold
    thresh = preprocessor.threshold(np.copy(smoothed), thresh_type)
    # thresh = threshold_local(smoothed, block_size = 31, offset=40)
    print('thresh is', thresh)
    im_max = smoothed.max()
    print('im_max is', im_max)
    binary = smoothed > thresh

    # Two approaches
    # 1. Identify local maxima in real-space image - separate by watershedding
    if sep_method == 'watershed':
        print('watershedding')
        local_maxi = peak_local_max(np.copy(smoothed), min_distance=min_radius, threshold_abs = thresh, indices=False, labels=np.copy(smoothed))
        labeled_image = ndimage.label(local_maxi, structure=square(3))[0]
        markers = ndimage.label(local_maxi)[0]
        labeled_image = watershed(-labeled_image, markers, mask=binary)

    # Calculate properties of particles
    # Properties - area, coords, label, radius
    properties = []
    columns = ('x', 'y', 'area', 'radius', 'intensity')
    indices = []
    # f_prop = regionprops(labeled_image, intensity_image = max_int_proj)
    f_prop = regionprops(labeled_image, intensity_image = max_int_proj)

    for d in f_prop:
        radius = (d.area / np.pi)**0.5
        properties.append([d.weighted_centroid[1], d.weighted_centroid[0], d.area, radius, d.mean_intensity*d.area])
        indices.append(d.label)
    # if not len(indices):
    #     all_props = pd.DataFrame([], index=[])
    indices = pd.Index(indices, name='label')
    properties = pd.DataFrame(properties, index=indices, columns=columns)
    properties = properties[properties['intensity'] < 10000]
    properties = properties[properties['intensity'] > mass_cutoff]
    properties = properties[properties['area'] > area_cutoff]
    properties['np_smooth_type'] = smooth_type
    properties['np_smooth_size'] = smooth_size
    properties['np_thresh_method'] = thresh_type
    properties['separation_method'] = sep_method
    particles_averaged = properties

# ################################
#     # Create data array in which clustered particles are averaged
#     # Store computed values
#     values = []
#     if len(properties) == 0:
#         values.append(['NaN', 'NaN', 'NaN', 'NaN', 'NaN'])
#     elif len(properties) > 0:
#         if len(properties) == 1:
#             properties['particle'] = 1
#         elif len(properties) > 1:
#             # Clustering to eliminate maxima that are too close together
#             positions = properties[['x', 'y']].values
#             # Distance matrix is n-particles x n-particles in size - reckon it gives the interparticle separation
#             # This gives the upper triangle of the distance matrix 
#             dist_mat = dist.pdist(positions)
#             link_mat = hier.linkage(dist_mat)
#             # fcluster assigns each of the particles in positions a cluster to which it belongs
#             cluster_idx = hier.fcluster(link_mat, max_radius, criterion='distance')
#             properties['particle'] = cluster_idx
            
#         particles = np.unique(properties['particle'].values)


#         for i in particles:
#             # Calculate weighted average position of particles
#             current = properties[properties['particle'] == i]
#             # Normalisation constant of weighted average
#             norm = np.sum(current['intensity'])
#             x_av = np.mean(current['x'])
#             y_av = np.mean(current['y'])
#             a_total = np.sum(current['area'])
#             # Geometric cut-off to particle size
#             if a_total > 2 and a_total < 400:
#                 values.append([x_av, y_av, a_total, norm, i])
        
    # # Data Frame containing the weighted averages of the locations of the particles
    # columns = ('x', 'y', 'a_total', 'intensity', 'particle')
    # particles_averaged = pd.DataFrame(values, columns = columns)
    # print 'Found', len(particles_averaged), 'particles'
    if test == True:
        vis.testplot_parts(max_int_proj, name_in, particles_averaged['x'], particles_averaged['y'], smoothed, binary, labeled_image)

    if plot == True:
        # Plot data
        plt.clf()
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.imshow(max_int_proj, cmap='gray', vmin=0, vmax=255)
        ax.scatter(particles_averaged['x'], particles_averaged['y'], facecolors='none', edgecolors='red', s=100)

    if save == True:
        outname = name_in + '_particles.tif'
        plt.savefig(outname, bbox_inches='tight')
        print(outname, 'saved')
        plt.close()

    return particles_averaged


def nps_finder_2d_stack(image_in, name_in, z_in=0, thresh_type='yen', smooth_type='mean', smooth_size=1, gauss_size=0, cont_size=0, sep_method='watershed', mass_cutoff=200, area_cutoff=0, max_radius=5, min_radius=1, test=True):
    # np.copy() used because peak_local_max does weird things to the image histogram if we use the actual image
    print('mass_cutoff is', mass_cutoff)
    y_size, x_size = np.shape(image_in)
    smoothed = np.copy(image_in)

    # Calculate threshold
    thresh = preprocessor.threshold(np.copy(smoothed), thresh_type)
    print('thresh is', thresh)
    im_max = smoothed.max()
    print('im_max is', im_max)
    binary = smoothed > thresh

    # Two approaches
    # 1. Identify local maxima in real-space image - separate by watershedding
    if sep_method == 'old_watershed':
        print('old_watershed')
        local_maxi = peak_local_max(np.copy(smoothed), min_distance=min_radius, threshold_abs = thresh, indices=False, labels=np.copy(smoothed))
        labeled_image = ndimage.label(local_maxi, structure=disk(1))[0]
        # markers = ndimage.label(local_maxi)[0]
        labeled_image = watershed(-labeled_image, labeled_image, mask=binary)
        # print(labeled_image)

    if sep_method == 'binary_label':
        print('binary_label')
        labeled_image = ndimage.label(binary, structure=disk(1))[0]
        # print(labeled_image)

    if sep_method == 'peak_label':
        print('peak_label')
        local_maxi = peak_local_max(np.copy(smoothed), min_distance=min_radius, threshold_abs = thresh, indices=False, labels=np.copy(smoothed))
        labeled_image = ndimage.label(local_maxi, structure=disk(1))[0]
        # print(labeled_image)

    columns = ('x', 'y', 'area', 'intensity')
    if len(np.where(labeled_image != 0)[0]) == 0:
        properties = pd.DataFrame([], columns=columns)
    if len(np.where(labeled_image != 0)[0]) > 0:
        f_prop = regionprops(labeled_image, intensity_image = smoothed)
        properties = []
        for d in f_prop:
            properties.append([d.weighted_centroid[1], d.weighted_centroid[0], d.area, d.mean_intensity*d.area]) 
        properties = pd.DataFrame(properties, columns=columns)

    if test == True:
        preprocessor.testplot_parts(image_in, name_in, z_in, properties['x'], properties['y'], smoothed, binary, labeled_image)

    return properties


def nps_finder_stack(stack_in, name_in, thresh_type, smooth_type, smooth_size=1, gauss_size=0, cont_size=0, sep_method='watershed', mass_cutoff=200, area_cutoff=0, max_radius=5, min_radius=1, test=True, plot=True, save=True):
    # np.copy() used because peak_local_max does weird things to the image histogram if we use the actual image
    print('mass_cutoff is', mass_cutoff)
    z_size, y_size, x_size = np.shape(stack_in)

    # Generate flatted image from stack
    max_int_proj = stack_in.max(axis=0)
    thresh_global = preprocessor.threshold(max_int_proj, 'mean')
    # Image to store labels
    labeled_image = np.zeros_like(max_int_proj)

    # Array containing particles in each slice and their properties
    columns = ('x', 'y', 'area', 'intensity', 'z')
    properties = []
    particles = pd.DataFrame(properties, columns=columns)
    for i in range(0, z_size): 
        print(i)
        smoothed = np.copy(stack_in[i])
        smoothed = preprocessor.smooth(smoothed, smooth_type, smooth_size)
        smoothed = preprocessor.bandpass(np.copy(smoothed), gauss_size) 
        smoothed = preprocessor.contrast(np.copy(smoothed), cont_size)

        if smoothed.max() > thresh_global:
            print('Global threshold is', thresh_global)
            print('Slice above noise floor')
            particles_slice = nps_finder_2d_stack(stack_in[i], name_in, i, thresh_type, smooth_type, smooth_size, gauss_size, cont_size, sep_method, mass_cutoff, area_cutoff, max_radius, min_radius, test=False)
            particles_slice['z'] = i
            particles = particles.append(particles_slice)
        if smoothed.max() < thresh_global:
            print('Slice below noise floor')

    # Second cut-off - discard particles with low area or intensity
    particles = particles[particles['intensity'] < 10000]
    particles = particles[particles['intensity'] > mass_cutoff]
    particles = particles[particles['area'] > area_cutoff]
    particles['np_smooth_type'] = smooth_type
    particles['np_smooth_size'] = smooth_size
    particles['np_thresh_method'] = thresh_type
    particles['separation_method'] = sep_method

    # Now use hierarchical clustering algorithm to link particles along z-axis
    print(len(particles), 'found before clustering')
    particles_clustered = preprocessor.clusterer(particles, max_radius)

    # Calculate the average properties of the clustered particles
    # Third cut-off happens here - remove particles of too limited extent in z
    particles_averaged = preprocessor.averager(particles_clustered, area_cutoff)
    print(len(particles_averaged), 'remaining after clustering')

    if test == True:
        plt.clf()
        preprocessor.testplot_parts_stack(stack_in, name_in, particles_averaged)

    if plot == True:
        # Plot data
        plt.clf()
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.imshow(max_int_proj, cmap='gray')
        ax.scatter(particles_averaged['x'], particles_averaged['y'], facecolors='none', edgecolors='red', s=100)

    if save == True:
        outname = name_in + '_particles.tif'
        plt.savefig(outname, bbox_inches='tight')
        print(outname, 'saved')
        plt.close()

    return particles_clustered, properties


def nucleus_finder_dt(stack_in, name_in, smooth_type, smooth_size, cont_size, thresh_type, erode, min_radius, max_radius, test, plot, save):
    print('smooth_size', smooth_size)
    print('cont_size', cont_size)
    print('thresh_type', thresh_type)
    print('erode', erode)
    print('min_radius', min_radius)
    print('max_radius', max_radius)
    print('test', test)
    print('save', save)

    # stack_in = stack_in.max(axis=0)
    # z_size=1
    z_size, y_size, x_size = int(np.shape(stack_in)[0]), int(np.shape(stack_in)[1]), int(np.shape(stack_in)[2])
    nuclei_count = []

    # Need to determine global threshold based on max intensity projetion that has
    # undergone same preprocessing as the stacks undergo
    max_int_proj = stack_in.max(axis=0)
    max_int_proj = preprocessor.smooth(np.copy(max_int_proj), smooth_type, smooth_size, cont_size)
    max_int_proj = preprocessor.contrast(np.copy(max_int_proj), cont_size)
    thresh = preprocessor.threshold(np.copy(max_int_proj), thresh_type)

    ####################################################################
    # Currently cluster in x and y then z separately - must fix this
    ####################################################################
    for im in range(0, z_size):
        image = stack_in[im]
        smoothed = preprocessor.smooth(np.copy(image), smooth_type, smooth_size, cont_size)
        smoothed = preprocessor.contrast(np.copy(smoothed), cont_size)
        # thresh = preprocessor.threshold(np.copy(smoothed), thresh_type)
        binary = smoothed > thresh
        # binary_closed = binary_closing(binary)
        binary_closed = binary_fill_holes(binary)

        if erode > 0:
            print('Performing', str(erode), 'binary erosions')
            eroded = binary_closed
            # n_erodes = min_radius // 2
            for i in range(0, erode):
                eroded = binary_erosion(eroded)
        elif erode == 0:
            eroded = binary_closed

        # Distance transform and threshold thereof
        distance = ndimage.distance_transform_edt(eroded)
        distance = distance > min_radius
        distance = ndimage.distance_transform_edt(distance)

        # Only look for local maxima if there's anything left after erosion:
        if len(np.where(distance != 0)[0]) == 0:
            print('Zero distance matrix')
            labeled = np.zeros_like(image)
            preprocessor.testplot(image, im, [], [], [], [], smoothed, binary_closed, eroded, distance, labeled)
            continue

        # local_maxi = peak_local_max(distance, min_radius, exclude_border=False, indices=False, labels=smoothed)
        local_maxi = peak_local_max(distance, exclude_border=False, indices=False, footprint=square(3), labels=smoothed)
        # local_maxi = local_maxima(distance)
        x, y = np.where(local_maxi != False)[1], np.where(local_maxi != False)[0]

        # List for storing x and y-coords on each slice
        x_slice, y_slice = [], []
        if len(x) == 0:
            print('No peaks detected')

        if len(x) == 1:
            print('Only one maximum detected')
            dist_non_zero = np.where(distance != 0)
            print(dist_non_zero)
            # Label the 1 remaining region
            labeled = distance
            labeled[dist_non_zero] = 1
            labeled = labeled.astype(int)
            print(labeled[np.where(labeled != 0)])
            f_prop = regionprops(labeled, intensity_image=image)
            for d in f_prop:
                print('len(f_prop)', len(f_prop))
                radius = (d.area / np.pi)**0.5
                nuclei_count.append([d.weighted_centroid[1], d.weighted_centroid[0], im, d.area, radius, d.mean_intensity*d.area])
                x_slice.append(d.weighted_centroid[1])
                y_slice.append(d.weighted_centroid[0])

            # Produce test plot
            preprocessor.testplot(image, im, x, y, x_slice, y_slice, smoothed, binary_closed, eroded, distance, labeled)

        if len(x) > 1:
            # Use hierarchical clustering algorithm to cluster maxima
            # Could alternatively use structure-based labelling
            print('Detected', str(len(x)), 'peaks')
            positions = np.stack((y, x), axis=1)
            # Distance matrix is n-particles x n-particles in size
            # This gives the upper triangle of the distance matrix 
            dist_mat = dist.pdist(positions)
            link_mat = hier.linkage(dist_mat)
            # fcluster assigns each of the particles in positions a cluster to which it belongs
            cluster_idx = hier.fcluster(link_mat, min_radius, criterion='distance')
            particles = np.unique(cluster_idx)

            markers = np.zeros_like(smoothed)
            markers[y, x] = cluster_idx

            # markers = ndimage.label(local_maxi, structure=square(3))[0]
            # Now that maxima have been clustered, label them by watershedding
            labeled = watershed(-distance, markers, mask=binary)

            f_prop = regionprops(labeled, intensity_image=image)
            for d in f_prop:
                radius = (d.area / np.pi)**0.5
                nuclei_count.append([d.weighted_centroid[1], d.weighted_centroid[0], im, d.area, radius, d.mean_intensity*d.area])
                x_slice.append(d.weighted_centroid[1])
                y_slice.append(d.weighted_centroid[0])
        #Plot binary and watershedded images showing identified peaks
        if test == True:
            vis.testplot(image, name_in, im, x, y, x_slice, y_slice, smoothed, binary_closed, eroded, distance, labeled)

    # Now we've scanned all the way through the stack and located things
    # we think are nuclei in each place. Now we need to cluster them in 3D so
    # we link together the slices
    values = []
    if len(nuclei_count) > 1:
        # Cluster nuclei
        columns = ('x', 'y', 'z' ,'area', 'radius', 'intensity')
        nuclei_count = pd.DataFrame(nuclei_count, columns=columns)

        # Now cluster nuclei by x, y coordinates
        positions = np.stack((nuclei_count['x'].values, nuclei_count['y'].values, nuclei_count['z'].values), axis=1)
        dist_mat = dist.pdist(positions)
        link_mat = hier.linkage(dist_mat)
        cluster_idx = hier.fcluster(link_mat, min_radius, criterion='distance')
        particles = np.unique(cluster_idx)

        # Calculate weighted average position of particles
        nuclei_count['particle'] = cluster_idx
        for j in particles:
            current = nuclei_count[nuclei_count['particle'] == j]
            # Normalisation constant of weighted average
            norm = np.sum(current['intensity'])
            x_av = np.sum(current['intensity'] * current['x'])/norm
            y_av = np.sum(current['intensity'] * current['y'])/norm
            z_av = np.sum(current['intensity'] * current['z'])/norm
            a_max = np.amax(current['area'])
            # Exclude nuclei on minimum intensity
            if norm > 10000:
                values.append([x_av, y_av, z_av, a_max, norm])

    # Data Frame containing the weighted averages of the locations of the particles
    columns = ('x', 'y', 'z', 'a_max', 'total_intensity')
    nuclei_averaged = pd.DataFrame(values, columns = columns)
    print('Found', len(nuclei_averaged), 'nuclei')

    if plot == True:
        # Numbers purely for visualisation purposes
        plt.clf()
        xy_scale, z_scale = 1., 2.
        # Plot 3D visualisation of data
        fig = plt.figure(figsize=(12, 12))
        # xy projection:
        ax_xy = fig.add_subplot(111)
        ax_xy.imshow(stack_in.max(axis=0), cmap='gray')
        ax_xy.scatter(nuclei_averaged['x'], nuclei_averaged['y'], facecolors='none', edgecolors='red', s=100)

        divider = make_axes_locatable(ax_xy)
        ax_zx = divider.append_axes("top", 2, pad=0.2, sharex=ax_xy)
        ax_zx.imshow(stack_in.max(axis=1), aspect=z_scale/xy_scale, cmap='gray')
        ax_zx.scatter(nuclei_averaged['x'], nuclei_averaged['z'], facecolors='none', edgecolors='red', s=100)
        ax_yz = divider.append_axes("right", 2, pad=0.2, sharey=ax_xy)
        ax_yz.imshow(stack_in.max(axis=2).T, aspect=xy_scale/z_scale, cmap='gray')
        ax_yz.scatter(nuclei_averaged['z'], nuclei_averaged['y'], facecolors='none', edgecolors='red', s=100)
        plt.draw()

    if save == True:
        outname = name_in + '_nuclei_py.tif'
        plt.savefig(outname, bbox_inches='tight')
        print(outname, 'saved')
        plt.close()

    return nuclei_averaged


# def nucleus_finder_2d(stack_in, name_in, smooth_type, smooth_size, cont_size, thresh_type, erode, min_radius, max_radius, test, plot, save):
#     print('smooth_size', smooth_size)
#     print('cont_size', cont_size)
#     print('thresh_type', thresh_type)
#     print('erode', erode)
#     print('min_radius', min_radius)
#     print('max_radius', max_radius)
#     print('test', test)
#     print('plot', plot)
#     print('save', save)

#     max_int_proj = stack_in.max(axis=0)
#     y_size, x_size = int(np.shape(stack_in)[0]), int(np.shape(stack_in)[1])
#     nuclei_count = []

#     ####################################################################
#     # Currently cluster in x and y then z separately - must fix this
#     ####################################################################

#     image = max_int_proj
#     smoothed = preprocessor.smooth_nuclei(np.copy(image), smooth_type, smooth_size, cont_size)
#     smoothed = preprocessor.contrast_nuclei(np.copy(smoothed), cont_size)
#     thresh = preprocessor.threshold_nuclei(np.copy(smoothed), thresh_type)
#     binary = smoothed > thresh
#     # binary_closed = binary_closing(binary)
#     binary_closed = binary_fill_holes(binary)

#     if erode > 0:
#         print('Performing', str(erode), 'binary erosions')
#         eroded = binary_closed
#         # n_erodes = min_radius // 2
#         for i in range(0, erode):
#             eroded = binary_erosion(eroded)
#     elif erode == 0:
#         eroded = binary_closed

#     # Distance transform and threshold thereof
#     distance = ndimage.distance_transform_edt(eroded)
#     distance = distance > min_radius
#     distance = ndimage.distance_transform_edt(distance)

#     # Only look for local maxima if there's anything left after erosion:
#     if len(np.where(distance != 0)[0]) == 0:
#         print('Zero distance matrix')
#         labeled = np.zeros_like(image)
#         preprocessor.testplot(image, name_in, [], [], [], [], smoothed, binary_closed, eroded, distance, labeled)

#     # local_maxi = peak_local_max(distance, min_radius, exclude_border=False, indices=False, labels=smoothed)
#     # local_maxi = peak_local_max(distance, exclude_border=False, indices=False, footprint=square(3), labels=smoothed)
#     local_maxi = local_maxima(distance)

#     # Differentiate local maxima by determining whether they 

#     x, y = np.where(local_maxi != False)[1], np.where(local_maxi != False)[0]

#     # List for storing x and y-coords on each slice
#     x_slice, y_slice = [], []
#     if len(x) == 0:
#         print('No peaks detected')

#     if len(x) == 1:
#         print('Only one maximum detected')
#         dist_non_zero = np.where(distance != 0)
#         print(dist_non_zero)
#         # Label the 1 remaining region
#         labeled = distance
#         labeled[dist_non_zero] = 1
#         labeled = labeled.astype(int)
#         print(labeled[np.where(labeled != 0)])
#         f_prop = regionprops(labeled, intensity_image=image)
#         for d in f_prop:
#             print('len(f_prop)', len(f_prop))
#             radius = (d.area / np.pi)**0.5
#             nuclei_count.append([d.weighted_centroid[1], d.weighted_centroid[0], d.area, radius, d.mean_intensity*d.area])
#             x_slice.append(d.weighted_centroid[1])
#             y_slice.append(d.weighted_centroid[0])

#         # Produce test plot
#         preprocessor.testplot(image, name_in, x, y, x_slice, y_slice, smoothed, binary_closed, eroded, distance, labeled)

#     if len(x) > 1:
#         print('Detected', str(len(x)), 'peaks')
#         # Cluster-based image segmentation
#         # positions = np.stack((y, x), axis=1)
#         # # Distance matrix is n-particles x n-particles in size
#         # # This gives the upper triangle of the distance matrix 
#         # dist_mat = dist.pdist(positions)
#         # link_mat = hier.linkage(dist_mat)
#         # # fcluster assigns each of the particles in positions a cluster to which it belongs
#         # cluster_idx = hier.fcluster(link_mat, min_radius, criterion='distance')
#         # particles = np.unique(cluster_idx)

#         # markers = np.zeros_like(smoothed)
#         # markers[y, x] = cluster_idx

#         # Structure-based image segmentation
#         markers = ndimage.label(local_maxi, structure=disk(1))[0]
#         # Now that maxima have been clustered, label them by watershedding
#         labeled = watershed(-distance, markers, mask=binary)

#         f_prop = regionprops(labeled, intensity_image=image)
#         for d in f_prop:
#             radius = (d.area / np.pi)**0.5
#             nuclei_count.append([d.weighted_centroid[1], d.weighted_centroid[0], d.area, radius, d.mean_intensity*d.area])
#             x_slice.append(d.weighted_centroid[1])
#             y_slice.append(d.weighted_centroid[0])
#     #Plot binary and watershedded images showing identified peaks
#     if test == True:
#         print(len(x_slice), len(y_slice))
#         preprocessor.testplot(image, name_in, x, y, x_slice, y_slice, smoothed, binary_closed, eroded, distance, labeled)

#     values = []
#     if len(nuclei_count) > 0:
#         columns = ('x', 'y' ,'area', 'radius', 'intensity')
#         nuclei_count = pd.DataFrame(nuclei_count, columns=columns)

#     # Data Frame containing the weighted averages of the locations of the particles
#     columns = ('x', 'y', 'a_max', 'total_intensity')
#     nuclei_averaged = nuclei_count #pd.DataFrame(values, columns = columns)
#     print('Found', len(nuclei_averaged), 'nuclei')


#     if plot == True:
#         # Numbers purely for visualisation purposes
#         plt.clf()
#         xy_scale, z_scale = 1., 2.
#         # Plot 3D visualisation of data
#         fig = plt.figure(figsize=(12, 12))
#         # xy projection:
#         ax_xy = fig.add_subplot(111)
#         ax_xy.imshow(stack_in.max(axis=0), cmap='gray')
#         ax_xy.scatter(nuclei_averaged['x'], nuclei_averaged['y'], facecolors='none', edgecolors='red', s=100)
#         plt.draw()

#     if save == True:
#         outname = name_in + '_nuclei.tif'
#         plt.savefig(outname, bbox_inches='tight')
#         print(outname, 'saved')
#         plt.close()

#     return nuclei_averaged



