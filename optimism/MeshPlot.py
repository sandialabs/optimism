import matplotlib.pyplot as plt
import numpy

def plot_mesh(mesh):
    fig, ax = plt.subplots(nrows=1)

    for tri in mesh.conns:
        coord = mesh.coords.take(tri,0)
        xvals = coord[:,0]
        yvals = coord[:,1]
        ax.tricontourf(xvals, yvals, xvals, levels=10, cmap="RdBu_r")
        ax.triplot(xvals, yvals, 'ko-')

    plt.show()

    
def plot_mesh_with_field(mesh, field, fast=False, direction=0, plotName='sol.png'):
    fig, ax = plt.subplots(nrows=1)

    if fast:
        cntr = ax.tricontourf(mesh.coords[:,0]+field[:,0], mesh.coords[:,1]+field[:,1], field[:,direction], levels=10, cmap="RdBu_r")
    else:
        minField = min(field[:,direction])
        maxField = max(field[:,direction])
        levels = numpy.linspace(minField, maxField, 11)
        for tri in mesh.conns:
            coord = mesh.coords.take(tri,0)
            disp = field.take(tri,0)
            xvals = coord[:,0] + disp[:,0]
            yvals = coord[:,1] + disp[:,1]
            fvals = disp[:,direction]
            ax.tricontourf(xvals, yvals, fvals, levels=levels, cmap="RdBu_r")
            ax.triplot(xvals, yvals, 'k-')
        cntr = ax.tricontourf(xvals, yvals, fvals, levels=levels, cmap="RdBu_r")
        
    ax.set_aspect('equal', adjustable='box')

    fig.colorbar(cntr,ax=ax)
    
    plt.savefig(plotName)
    plt.close()

