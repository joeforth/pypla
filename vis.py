import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imagehist(image_in):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.hist(np.array(image_in).flatten(), log=True, bins=range(256))
    _ = ax.set_title('Min value: %i \n' 'Max value: %i \n' 'Image shape: %s \n' % (image_in.min(), image_in.max(), image_in.shape))


def testplot(image_in, name_in, z_in, x_peaks, y_peaks, x_labels, y_labels, smoothed_in, binary_in, eroded_in, distance_in, labeled_in):
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    axes[0,0].imshow(image_in, cmap='gray', vmin=0, vmax=255)
    axes[0,0].scatter(x_labels, y_labels, s=50, facecolor='none', edgecolor='red')
    axes[0,1].imshow(smoothed_in, cmap='gray', vmin=0, vmax=255)
    axes[0,2].imshow(binary_in)
    axes[1,0].imshow(eroded_in)
    axes[1,1].imshow(distance_in)
    axes[1,2].imshow(labeled_in)
    axes[1,2].scatter(x_peaks, y_peaks, s = 50, facecolor='none', edgecolor='red')

    axes[0,0].set_title('Raw Image')
    axes[0,1].set_title('Smoothed Image')
    axes[0,2].set_title('Binary Image')
    axes[1,0].set_title('Eroded Image')
    axes[1,1].set_title('Distance Transform')
    axes[1,2].set_title('Labeled Image')

    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    # plt.savefig('./test_data/' + name_in + '_slice_' + str(z_in) + '.tif', bbox_inches='tight')
    plt.close()


def testplot_parts(image_in, name_in, x_in, y_in, smoothed_in, binary_in, labeled_in):
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    axes[0,0].imshow(image_in, cmap='gray', vmin=0, vmax=255)
    axes[0,1].imshow(smoothed_in, cmap='gray', vmin=0, vmax=255)
    axes[0,1].scatter(x_in, y_in, s=50, facecolor='none', edgecolor='red')
    axes[0,2].imshow(binary_in)
    axes[1,0].imshow(labeled_in)
    axes[1,0].scatter(x_in, y_in, s = 50, facecolor='none', edgecolor='red')

    axes[0,0].set_title('Raw Image')
    axes[0,1].set_title('Smoothed Image')
    axes[0,2].set_title('Binary Image')
    axes[1,0].set_title('Labeled Image')

    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    # plt.show()
    plt.savefig(name_in + '_particles.tif', bbox_inches='tight')
    plt.close()


def stack_plotter(stack_in):
    z_size, y_size, x_size = np.shape(stack_in)

    nrows = np.int(np.ceil(np.sqrt(z_size)))
    ncols = np.int(z_size // nrows + 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    for n in range(z_size):
        i = n // ncols
        j = n % ncols
        axes[i, j].imshow(stack_in[n], interpolation='nearest', cmap='gray', vmin=0, vmax=255)
        
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    plt.draw()


def testplot_parts_3d(image_in, name_in, z_in, x_in, y_in, smoothed_in, binary_in, labeled_in):
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    axes[0,0].imshow(image_in, cmap='gray', vmin=0, vmax=255)
    axes[0,1].imshow(smoothed_in, cmap='gray', vmin=0, vmax=255)
    axes[0,1].scatter(x_in, y_in, s=50, facecolor='none', edgecolor='red')
    axes[0,2].imshow(binary_in)
    axes[1,0].imshow(labeled_in)
    axes[1,0].scatter(x_in, y_in, s = 50, facecolor='none', edgecolor='red')

    axes[0,0].set_title('Raw Image')
    axes[0,1].set_title('Smoothed Image')
    axes[0,2].set_title('Binary Image')
    axes[1,0].set_title('Labeled Image')

    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    # plt.show()
    plt.savefig(name_in + '_' + str(z_in) + '.tif', bbox_inches='tight')
    plt.close()


def testplot_parts_stack(stack_in, name_in, part_locs):
    n_rows = 1
    n_cols = 2
    print(part_locs)
    z_size, y_size, x_size = np.shape(stack_in)

    nrows = np.int(np.ceil(np.sqrt(z_size)))
    ncols = np.int(z_size // nrows + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    for n in range(z_size):
        # current = part_locs[part_locs['z'] < n + 1]
        # current = current[current['z'] >= n]
        current = part_locs[part_locs['frame'] == n]
        i = n // ncols
        j = n % ncols
        axes[i, j].imshow(stack_in[n], interpolation='nearest', cmap='gray', vmin=0, vmax=255)
        axes[i, j].scatter(current['x'], current['y'])
        
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    plt.draw()

    plt.savefig(name_in +'_slice.tif', bbox_inches='tight')
    plt.close()


def testplot_parts_stack_vid(stack_in, name_in, part_locs):
    n_rows = 1
    n_cols = 2
    print(part_locs)
    z_size, y_size, x_size = np.shape(stack_in)


    for n in range(0, z_size):
        current = part_locs[part_locs['z'] == n]
        plt.clf()
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(212)
        ax.imshow(stack_in[n], interpolation='nearest', cmap='gray', vmin=0, vmax=255)
        ax.scatter(current['x'], current['y'], facecolors='none', edgecolors='red', s=50)
        ax.set_xlim(0,512)
        ax.set_ylim(0,512)
        plt.savefig('./' + name_in +'_' + str(n) + '.tif', dpi=300, bbox_inches='tight')
        plt.close()


def particle_plotter_3d(stack_in, particles_in, name_in, dir_in='', save=True):
    # Numbers purely for visualisation purposes
    plt.clf()
    xy_scale, z_scale = 1., 2.
    # Plot 3D visualisation of data
    fig = plt.figure(figsize=(12, 12))
    # xy projection:
    ax_xy = fig.add_subplot(111)
    ax_xy.imshow(stack_in.max(axis=0), cmap='gray', vmin=0, vmax=255)
    ax_xy.scatter(particles_in['x'], particles_in['y'], facecolors='none', edgecolors='red', s=100, vmin=0, vmax=255)

    divider = make_axes_locatable(ax_xy)
    ax_zx = divider.append_axes("top", 2, pad=0.2, sharex=ax_xy)
    ax_zx.imshow(stack_in.max(axis=1), aspect=z_scale/xy_scale, cmap='gray', vmin=0, vmax=255)
    ax_zx.scatter(particles_in['x'], particles_in['z'], facecolors='none', edgecolors='red', s=100)
    ax_yz = divider.append_axes("right", 2, pad=0.2, sharey=ax_xy)
    ax_yz.imshow(stack_in.max(axis=2).T, aspect=xy_scale/z_scale, cmap='gray', vmin=0, vmax=255)
    ax_yz.scatter(particles_in['z'], particles_in['y'], facecolors='none', edgecolors='red', s=100)
    if save==True:
        plt.savefig(dir_in + name_in + '_particles.tif', bbox_inches='tight')
    elif save==False:
        plt.show()    
    plt.close()


# def nucleus_plotter(stack_in, nuclei_in, name_in, dir_in, save=True):
#     plt.clf()
#     fig = plt.figure(figsize=(12,12))
#     ax = fig.add_subplot(111)
#     ax.imshow(stack_in.max(axis=0), cmap='gray', vmin=0, vmax=255)
#     ax.scatter(nuclei_in['X'], nuclei_in['Y'], facecolors='none', edgecolors='red', s=100, vmin=0, vmax=255)
#     if save==True:
#         plt.savefig(dir_in + name_in + '_nuclei.tif', bbox_inches='tight')
#     elif save==False:
#         plt.show()
#     plt.close()


def nuclei_plotter(stack_in, dir_in, name_in, area_cutoff=0, plot=True, save=True):
#   Reads in nuclei locations found using ImageJ - outputs plot of nucleus locations
#   Also counts nuclei per image and accounts for nucleus location at edge of image
    z_size, y_size, x_size = np.shape(stack_in)
    nuc_locs = pd.read_csv(dir_in + name_in + '.csv', index_col=0)
    nuc_locs['edge'] = 0
    for i in range(0, len(nuc_locs)):
        current = nuc_locs.iloc[i]
    # Locate nuclei on top edge of image
        if current['BY'] == 0:
            current['edge'] = 1
    # Locate nuclei on bottom edge of image
        if current['BY'] + current['Height'] >= y_size:
            current['edge'] = 1
    # Locate nuclei on right edge of image
        if current['BX'] + current['Width'] >= x_size:
            current['edge'] = 1
    # Locate nuclei on left edge of image
        if current['BX'] == 0:
            current['edge'] = 1
        nuc_locs.iloc[i] = current

    # print('nuc_area_cutoff is', str(area_cutoff))
    nuc_locs = nuc_locs[nuc_locs['Area'] > area_cutoff]

    if plot == True:
        # Numbers purely for visualisation purposes
        plt.clf()
        # Plot 3D visualisation of data
        fig = plt.figure(figsize=(12, 12))
        # xy projection:
        ax_xy = fig.add_subplot(111)
        ax_xy.imshow(stack_in.max(axis=0), cmap='gray', vmin=0, vmax=255)
        bulk_nucs = nuc_locs[nuc_locs['edge'] == 0]
        ax_xy.scatter(bulk_nucs['X'], bulk_nucs['Y'], facecolors='none', edgecolors='green', s=100)
        edge_nucs = nuc_locs[nuc_locs['edge'] == 1]
        ax_xy.scatter(edge_nucs['X'], edge_nucs['Y'], facecolors='none', edgecolors='red', s=100)
        plt.draw()

    if save == True:
        outname = dir_in + '/' + name_in + '_nuclei.tif'
        plt.savefig(outname, bbox_inches='tight')
        # print(outname, 'saved')
        plt.close()

    return nuc_locs

