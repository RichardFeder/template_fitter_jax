import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
# from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np
from matplotlib import cm
# from PIL import Image
import glob
import config
import matplotlib.patches as mpatches


def plot_fraction_and_hist(zmag, dzoneplusz, zout,
                           zmin_list, colors_zmin,
                           mag_min=18, mag_max=24, nbins=30,
                           sigma_thresh=0.2,
                           outfilename=None, figsize=(8, 4), hist_lw=2, 
                          bbox_to_anchor=[0.1, 1.5], 
                          height_ratios=[1.5, 1]):
    """
    Create 2-panel plot:
    Top: fraction vs z-band mag for full sample and subsamples
    Bottom: histogram of z-band mags for counts in each bin.
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios': height_ratios})
    
    # ===== TOP PANEL =====
    # Full sample fraction vs mag
    bin_centers, fractions_all, counts_all, which_in_bin = fraction_low_sigma_per_mag_bin(
        zmag, dzoneplusz, mag_min=mag_min, mag_max=mag_max,
        nbins=nbins, sigma_thresh=sigma_thresh
    )
    ax1.plot(bin_centers, fractions_all, label='All',
             color='k', linestyle='dashed', linewidth=2)
    
    # Subsamples by zmin
    for zidx, zmin in enumerate(zmin_list):
        sel = zout > zmin
        bin_centers, fractions_vs_z, counts_vs_z, _ = fraction_low_sigma_per_mag_bin(
            zmag[sel], dzoneplusz[sel], mag_min=mag_min, mag_max=mag_max,
            nbins=nbins, sigma_thresh=sigma_thresh
        )
        ax1.plot(bin_centers, fractions_vs_z, marker='o',
                 label=r'$\hat{z}_{star} > %s$' % zmin,
                 color=colors_zmin[zidx])
    
    ax1.set_yscale('log')
    ax1.set_title(r'Fraction of stars with $\frac{\sigma_z}{1+\hat{z}} < '+str(sigma_thresh)+'$', fontsize=14)
    ax1.set_ylim(5e-4, 1.05)
    ax1.set_xlim(mag_min, mag_max)
    ax1.grid(True)
    ax1.legend(loc=2, fontsize=12, bbox_to_anchor=bbox_to_anchor, ncol=3)
    
    # ===== BOTTOM PANEL =====
    bins_hist = np.linspace(mag_min, mag_max, nbins+1)
    
    # Full sample histogram
    ax2.hist(zmag[which_in_bin], bins=bins_hist, histtype='stepfilled', color='k', linewidth=2,
             label='$\frac{\sigma_z}{1+\hat{z}} < '+str(sigma_thresh)+'$', alpha=0.2)
    
    ax2.hist(zmag, bins=bins_hist, histtype='step', color='k', linewidth=1.5,
             label='All')
    
    # Subsample histograms
    for zidx, zmin in enumerate(zmin_list):
        sel = (zout > zmin)
        
        fullsel = sel*(which_in_bin)
        ax2.hist(zmag[sel], bins=bins_hist, histtype='step',
                 color=colors_zmin[zidx], linewidth=hist_lw)
    
        
        ax2.hist(zmag[fullsel], bins=bins_hist, histtype='stepfilled', alpha=0.7,
                 color=colors_zmin[zidx], label=r'$\hat{z}_{star} > %s$' % zmin, linewidth=hist_lw)
    
    ax2.set_xlabel("z-band magnitude [AB]", fontsize=14)
    ax2.set_ylabel("Counts per mag bin", fontsize=12)
    ax2.grid(True)
    ax2.set_yscale('log')
    ax2.set_ylim(1, 1e4)
#     ax2.legend(fontsize=12)
    
#     fig.tight_layout()
    
    if outfilename:
        fig.savefig(outfilename, bbox_inches='tight', dpi=200)
    
    plt.show()
    
    return fig

def plot_map(image, figsize=(8,8), title=None, titlefontsize=16, xlabel='x [pix]', ylabel='y [pix]',\
             x0=None, x1=None, y0=None, y1=None, lopct=5, hipct=99,\
             return_fig=False, show=True, nanpct=True, cl2d=False, cmap='viridis', noxticks=False, noyticks=False, \
             cbar_label=None, norm=None, vmin=None, vmax=None, scatter_xs=None, scatter_ys=None, scatter_marker='x', scatter_color='r', \
             interpolation='none', cbar_fontsize=14, xylabel_fontsize=16, tick_fontsize=14, \
             textstr=None, text_xpos=None, text_ypos=None, bbox_dict=None, text_fontsize=16, origin='lower'):

    f = plt.figure(figsize=figsize)



    if vmin is None:
        vmin = np.nanpercentile(image, lopct)
    if vmax is None:
        vmax = np.nanpercentile(image, hipct)

    if title is not None:
    
        plt.title(title, fontsize=titlefontsize)

    print('min max of image in plot map are ', np.min(image), np.max(image))
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation, origin=origin, norm=norm)

    # if nanpct:
    #     plt.imshow(image, vmin=np.nanpercentile(image, lopct), vmax=np.nanpercentile(image, hipct), cmap=cmap, interpolation='None', origin='lower', norm=norm)
    # else:
    #     plt.imshow(image, cmap=cmap, origin='lower', interpolation='none', norm=norm)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=cbar_fontsize)

    if scatter_xs is not None and scatter_ys is not None:
        plt.scatter(scatter_xs, scatter_ys, marker=scatter_marker, color=scatter_color)
    if x0 is not None and x1 is not None:
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        
    if cl2d:
        plt.xlabel('$\\ell_x$', fontsize=xylabel_fontsize)
        plt.ylabel('$\\ell_y$', fontsize=xylabel_fontsize)
    else:
        plt.xlabel(xlabel, fontsize=xylabel_fontsize)
        plt.ylabel(ylabel, fontsize=xylabel_fontsize)

    if noxticks:
        plt.xticks([], [])
    if noyticks:
        plt.yticks([], [])

    if textstr is not None:
        if bbox_dict is None:
            bbox_dict = dict({'facecolor':'white', 'edgecolor':'None', 'alpha':0.7})
        plt.text(text_xpos, text_ypos, textstr, fontsize=text_fontsize, bbox=bbox_dict)
        
    plt.tick_params(labelsize=tick_fontsize)

    if show:
        plt.show()
    if return_fig:
        return f
