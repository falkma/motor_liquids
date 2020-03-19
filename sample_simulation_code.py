import hoomd
import hoomd.md
import numpy as np
import scipy.spatial
import gsd.hoomd
import os

hoomd.context.initialize("");

#the first part is all about initializing the simulation

#number of polymers
Npol = 1

#length of each polymer
Lpol = 15

#number of nanostar parcels
Nstar = 2400

#temperature
T = 1

#we are going to initialize as a grid with particle centers spaced apart
spacing = 1.02

#how long to simulate
step_num = 400000

#name of file to print to
folder = 'stability1'

#filling in positions
#first half on the left second on the right
#whole box will be 2*40 long I should probably not hard code everything
star_positions = np.zeros((Nstar,3))
for i in np.arange(Nstar//2):
    star_positions[i] = [spacing*(i%40), spacing*(i//40), 0]
for i in np.arange(Nstar//2):
    star_positions[Nstar//2+i] = [spacing*(40+i%40), spacing*(i//40), 0]

#initializing polymer perpendicular to interface
polymer_positions = [[spacing*40-Lpol/2+i,21,0] for i in np.arange(Lpol)]

positions = np.concatenate((star_positions,polymer_positions))

#need to mean center
polymer_positions -= np.mean(positions,axis = 0)
star_positions -= np.mean(positions, axis = 0)
positions = np.concatenate((star_positions,polymer_positions))

#we simulate with periodic boundary conditions and you can see where lx and ly get turned into the boxdims below
lx = np.max(np.abs(positions[:,0]))
ly = np.max(np.abs(positions[:,1]))

#initalizing simulation
snapshot = hoomd.data.make_snapshot(N=Npol*Lpol+Nstar,  box=hoomd.data.boxdim(Lx=2*(lx+1), Ly=2*(ly+1), Lz=1,dimensions=2),particle_types=['A','B','P'],bond_types=['polymer'],angle_types=['polymer'])
snapshot.particles.position[:] =  positions

#this is where we assign particle types
#every particle gets a type which is how hoomd keeps track of what interactions it experiences
# 0 for A 1 for B 2 for polymer
snapshot.particles.typeid[0:(Nstar)//2]=0
snapshot.particles.typeid[(Nstar)//2:Nstar]=1
snapshot.particles.typeid[Nstar:]=2

#now we need to tell hoomd that there will be bonds between adjacent polymer particles
snapshot.bonds.resize(Lpol-1)
snapshot.bonds.group[:] = [[i,i+1] for i in Nstar+np.arange(Lpol-1)]
snapshot.bonds.typeid[:] = 0

#now we need to tell hoomd that there will be a bending force between consecutive triples of monomers
snapshot.angles.resize(Lpol-2)
snapshot.angles.group[:] = [[i,i+1,i+2] for i in Nstar+np.arange(Lpol-2)]
snapshot.angles.typeid[:] = 0

#you can tile your system an arbitrary number of times if you want
xrep = 1
yrep = 1
N = xrep*yrep*(Lpol+Nstar)
Npol = Npol*xrep*yrep
Nstar = Nstar*xrep*yrep
snapshot.replicate(xrep,yrep,1)

#now we are starting to add physics to the simulation
#initializing the simulation object
system = hoomd.init.read_snapshot(snapshot);

#initalizing cell list
nl = hoomd.md.nlist.cell()

#harmonic bonds for the polymer
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k=50.0*T, r0=1)

#harmonic angle penalty
lp = 90
stiffness = hoomd.md.angle.harmonic()
stiffness.angle_coeff.set('polymer', k=lp*T, t0=3.142)

#morse potential - you can look up the details, but D0 is an overall multiplicative factor
#affecting depth of the well, but also the strength of the repulsion
#setting D0 to zero renders the particles transparent to each other
morse = hoomd.md.pair.morse(r_cut=2.0, nlist=nl)
morse.pair_coeff.set('A', 'A', D0=6*T, alpha=2, r0=1.2)
morse.pair_coeff.set('A', 'B', D0=2*T, alpha=2, r0=1.2)
morse.pair_coeff.set('B', 'B', D0=6*T, alpha=2, r0=1.2)
morse.pair_coeff.set('A', 'P', D0=0*T, alpha=2, r0=1.2)
morse.pair_coeff.set('B', 'P', D0=0*T, alpha=2, r0=1.2)
morse.pair_coeff.set('P', 'P', D0=0*T, alpha=2, r0=1.2)

all = hoomd.group.all();

#defining some global variables which the activity update function will inherit
pol_points = np.where(snapshot.particles.typeid[:] == 2)[0]
pol_points = pol_points[1:-1]

#assigning the polarity of motors
motor_type = np.array([ -1 if snapshot.particles.typeid[i] == 0 else 1 if snapshot.particles.typeid[i] == 1 else 0 for i in np.arange(N) ])

#cutoff and magnitude of the motor force
cutoff = 1.2
f_mag = .1

def activity_update(timestep):

    if timestep%5 == 0:

        config = system.take_snapshot()
        positions = config.particles.position

        #bond lengths so we can compute the tangent along each polymer
        bond_lengths = np.linalg.norm(positions[1:]-positions[0:-1], axis = 1)

        #compute distances between all particles and the polymer particles
        motor_log = scipy.spatial.distance.cdist(positions,positions[pol_points])

        #reset all motor forces to zero
        active.set_force(fvec=(0,0,0),group=all)

        for idx,i in enumerate(pol_points[::2]):

            #find indices of motor particles which are within cutoff of the monomer
            motors = np.where(motor_log[:,idx] < cutoff)[0]
            motors = motors[np.where(np.abs(motor_type[motors]) > 0)[0]]

            if motors.size > 0 :

                #estimate tangent at the monomer
                t_vec = (positions[i]-positions[i-1])/(bond_lengths[i-1]+.000001)

                #compute and set force vector for the monomer
                f_pol = f_mag*t_vec*np.sum(motor_type[motors])
                active.set_force(tag=i, fvec=(f_pol[0],f_pol[1],0))

                #assign the equal and opposite force to the motor particle
                for m in motors:
                    f_motor = -f_mag*t_vec*motor_type[m]
                    active.set_force(tag=m , fvec=(f_motor[0],f_motor[1],0))

    return 1

#add motor force
active = hoomd.md.force.constant(callback=activity_update)


#this is where we start to think about dynamics and actually run the simulation
#set hyper parameters of simulation
hoomd.md.integrate.mode_standard(dt=0.005);

#set integration method
hoomd.md.integrate.brownian(group=all, kT=T, dscale = 5,seed=123);

#indicate where and what data should be retained
hoomd.dump.gsd(folder+'.gsd', period=500, group=all, overwrite=True, dynamic = ['property','momentum']);

#run the simulation
hoomd.run(step_num)
