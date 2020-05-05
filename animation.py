import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation


class animator:
    def __init__(self, X, Y, Z, sequence,
            fig=0, name=None, interval=1):
        self.fig = plt.figure(fig)
        ax = self.fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none', alpha=0.75)

        self.top = np.max(Z) + np.abs(np.min(Z))
        self.bottom = np.min(Z) - np.abs(np.max(Z))
        cset = ax.contourf(X,Y,Z,zdir='z',offset=self.bottom,cmap=cm.coolwarm)
        ax.set_zlim(self.bottom, self.top)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if name != None:
            ax.title.set_text(name)
    
        self.origin = ax.scatter3D([],[],[],color='red',marker='*')
        self.projected = ax.scatter3D([],[],[],color='black')

        self.X = X
        self.Y = Y
        self.Z = Z
        self.interval = interval
        self.sequence = sequence


    def animate(self,i):
        x,y = self.sequence[i]
        self.origin._offsets3d = (self.X[x,y], self.Y[x,y], self.Z[x,y])
        self.projected._offsets3d = (self.X[x,y], self.Y[x,y], self.bottom)

    def render(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate,
                frames=np.arange(len(self.sequence)), interval=self.interval, blit=False, repeat=False)
        plt.show()
