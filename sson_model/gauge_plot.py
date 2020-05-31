import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle
import numpy
from PIL import Image
import pathlib

def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, title=''): 
    
    """
    main gauge function
    """
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))
 
    
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(numpy.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    
    fig, ax = plt.subplots(figsize=(8.5,6.5))#plt.subplots(figsize=(5,2.5))

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    
    [ax.add_patch(p) for p in patches]

    
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    as well as markers
    """
    markers_del_n = 1./N 
    range_markers = numpy.arange(0, 1+markers_del_n,markers_del_n)[::-1]
    range_marker_loc = numpy.append(ang_range[:,0],ang_range[-1,1]) 

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.35 * numpy.cos(numpy.radians(mid)), 0.35 * numpy.sin(numpy.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=12, \
            fontweight='bold', rotation = rot_text(mid))
    
    for start, mark in zip(range_marker_loc, range_markers):
        ax.text(0.4 * numpy.cos(numpy.radians(start)), 0.4 * numpy.sin(numpy.radians(start)), mark, \
            horizontalalignment='center', verticalalignment='center', fontsize=8, \
            fontweight='bold', rotation = rot_text(start))

    """
    set the bottom banner and the title
    """
    # r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    # ax.add_patch(r)
    
    # ax.text(0, -0.05, title, horizontalalignment='center', \
    #      verticalalignment='center', fontsize=12, fontweight='bold')

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax.arrow(0, 0, 0.225 * numpy.cos(numpy.radians(pos)), 0.225 * numpy.sin(numpy.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')

    # Now add quiet and ss ovals on left and right corners of the plot 
    if pathlib.Path("../misc").is_dir():
        quiet_oval = "../misc/quiet_oval1.png"
        ss_oval = "../misc/ss_oval1.png"
    else:
        quiet_oval = "./misc/quiet_oval1.png"
        ss_oval = "./misc/ss_oval1.png"

    im_quiet = Image.open(quiet_oval)
    height_quiet = im_quiet.size[1]
    im_ss = Image.open(ss_oval).resize(\
                        (im_quiet.size[0],im_quiet.size[1]),\
                        Image.ANTIALIAS)
    height_ss = im_ss.size[1]

    fig.figimage(im_quiet, 0, fig.bbox.ymax + height_quiet, zorder=1)
    fig.figimage(im_ss, 1400, fig.bbox.ymax + height_quiet, zorder=1)


    plt.tight_layout()
    return fig

def degree_range(n): 
        # helper func for plotting a gauge plot
        start = numpy.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = numpy.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return numpy.c_[start, end], mid_points

def rot_text(ang): 
    # helper func for plotting a gauge plot
    rotation = numpy.degrees(numpy.radians(ang) * numpy.pi / numpy.pi - numpy.radians(90))
    return rotation