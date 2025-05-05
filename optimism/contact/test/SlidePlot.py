from matplotlib import pyplot
from matplotlib.widgets import Slider
import numpy as np

class slide_plot():
    def __init__(self, fig, axs, time_range):
        self.axs = axs
        self.time_range = time_range
        fig.subplots_adjust(bottom=0.25)
        axn = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(axn, 'time', self.time_range[0], self.time_range[-1], valinit=self.time_range[0], valstep=time_range)
        self.lines = []
        self.patches = []
    
    def plot(self, frame, plot_func_of_t, mods='k'):
        x,y = plot_func_of_t(self.time_range[0])
        line, = self.axs[frame].plot(x,y,mods)
        self.lines.append( (line, plot_func_of_t) )

    def arrow(self, frame, plot_func_of_t):
        x,y,dx,dy = plot_func_of_t(self.time_range[0])
        print(x,y,dx,dy)
        N = len(x)
        self.arrow_scaling = 0.05
        colors = 'r','g','r--','g--'
        for n in range(N):
            patch = self.axs[frame].arrow(x[n],y[n],self.arrow_scaling*dx[n],self.arrow_scaling*dy[n], color=colors[n], width=0.01)
            self.patches.append( (patch, n, plot_func_of_t) )

    def show(self):

        def update(t):
            t = self.slider.val
            for line, func in self.lines:
                x,y = func(t)
                line.set_xdata(x)
                line.set_ydata(y)

            for patch, n, func in self.patches:
                x,y,dx,dy = func(t)
                patch.set_data(x=x[n],y=y[n],dx=self.arrow_scaling*dx[n],dy=self.arrow_scaling*dy[n])

        self.slider.on_changed(update)

        pyplot.show()
    