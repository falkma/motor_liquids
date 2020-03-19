import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
import os

#identify simulation name
folder = 'network4'

#how long the simulation was
step_num = 400000

#how often hoomd wrote to the gsd file
frame_num = step_num/500

#only take every fourth frame
frame_skip = 4

#extract all frames
frames = dict()
traj = gsd.hoomd.open(folder+'.gsd')
for i in np.arange(frame_num):
    frames[str(i)] = traj.read_frame(i).particles.position

#extract particle ids
c = traj.read_frame(0).particles.typeid[:]
cmax = np.nanmax(c)

#import colormaps
cms = matplotlib.cm

files = []

#set up the figure dimensions
fig = plt.figure(figsize=(8,10))
lx = 1*np.max(np.abs(traj.read_frame(0).particles.position[:,0]))
ly = 1*np.max(np.abs(traj.read_frame(0).particles.position[:,1]))

#extract size of simulation
N = len(traj.read_frame(0).particles.position)

#if you dont want to start at the first frame you can indicate that here
lag = 10

for i in np.arange((frame_num-lag)//frame_skip):
    ax = fig.add_subplot(111, aspect='equal')
    ax.axis([-lx, lx, -ly, ly])
    ax.axis('off')
    for x, y, r, color in zip(frames[str(frame_skip*i+lag)][:,0], frames[str(frame_skip*i+lag)][:,1], .5*np.ones(N),c*255/cmax):
        ax.add_artist(Circle(xy=(x, y),radius=r,color = cms.brg(color)))
    fname = '_tracer_tmp%03d.png' % i
    files.append(fname)
    plt.savefig(fname)
    plt.clf()
    print(i)

if os.path.exists(folder+'_tracer_movie.mp4'):
    os.system('rm '+folder+'_tracer_movie.mp4')

#making the movie
os.system("ffmpeg -i _tracer_tmp%03d.png -c:v libx264 -pix_fmt yuv420p -framerate 1/5 -r 30 "+folder+"_tracer_movie.mp4")

#cleaning up the mess from all the temporary files that were made
for fname in files:
    os.remove(fname)
