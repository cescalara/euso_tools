import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib

# Make function to plot
def plot_focal_surface(focal_surface, threshold = 0):
    # set font size
    matplotlib.rcParams.update({'font.size': 22})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if threshold != 0:
        p = plt.imshow(focal_surface, axes=ax,
                       interpolation='nearest', origin = 'lower',
                       vmin = 0, vmax = threshold)
    else:
        p = plt.imshow(focal_surface, axes=ax,
                       interpolation='nearest', origin = 'lower', cmap = cm.YlGnBu_r)
        
    plt.xlabel('pixel X')
    plt.ylabel('pixel Y')

    # Set ticks
    major_ticks = np.arange(-.5, 48, 8)
    minor_ticks = np.arange(-.5, 48, 1)
    #ax.set_xlim(0.0,47.0)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xticklabels(np.arange(0, 49, 8));
    ax.set_yticklabels(np.arange(0, 49, 8));

    # Set grid
    #ax.grid(which='minor', alpha=0.2)
    ax.grid(color='k', linestyle='-', linewidth=2)
    ax.grid(which='major', alpha=0.4)

    # Add colourbar
    fig.subplots_adjust(right=1.0)
    cbar_ax = fig.add_axes([1.0, 0.1, 0.05, 0.8])
    fmt = mpl.ticker.LogFormatterSciNotation()
    cbar=fig.colorbar(p, cax=cbar_ax)
    cbar.set_label('counts', x = 1.2)
    cbar.formatter.set_powerlimits((0, 0))
    #cbar.set_clim(0, np.max(focal_surface))

    return p


# not working yet
def anim_pdm(focal_surface_packet, gtu_num, gtu_range, threshold = 0):
    """
    animate the pdm over gtu_range
    """
    from matplotlib import animation, rc
    from IPython.display import HTML
    rc('animation', html='html5')
    
    fig = plt.figure(figsize = (10, 10))
        
    # inital frame
    init_frame = focal_surface_packet[gtu_num]
    if threshold == 0:
        im = plt.imshow(init_frame, cmap = 'viridis', animated = True)
    else:
        im = plt.imshow(init_frame, cmap = 'viridis', animated = True, vmax = threshold)

    gtu_num = 0
    def updatefig(*args):
        global frame, gtu_num
        gtu_num += 1
        frame = focal_surface_packet[gtu_num]
        im.set_array(frame)
        return im,
    
    # animation
    anim = animation.FuncAnimation(fig, updatefig,
                                       frames = gtu_range, interval = 500, blit = True)
    HTML(anim.to_html5_video())

