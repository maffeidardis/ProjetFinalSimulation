import ParticleClass_COVID as pc
import numpy as np
import time as tempo

import os
from matplotlib import cm
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#np.random.seed(999)
Snapshot_output_dir = './SnapshotsMonomers'
if not os.path.exists(Snapshot_output_dir): os.makedirs(Snapshot_output_dir)


'''Initialize system with following parameters'''
NumberOfMonomers = 150
L_xMin, L_xMax = 0, 100
L_yMin, L_yMax = 0, 50
NumberMono_per_kind = np.array([NumberOfMonomers])
Radiai_per_kind = np.array([ 0.5 ])
Densities_per_kind = np.array([ 1.0])
k_BT = 5

#initalising covid related parameters:
    
NumberOfCovid = 1
p_covid = 0.5
covid_duration = 0.02 * 500 #initialized as mutlple of the dt_frame

# call constructor, which should initialize the configuration
mols = pc.patient(NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT, NumberOfCovid, p_covid, covid_duration)

    
mols.snapshot( FileName = Snapshot_output_dir+'/InitialConf.png', Title = '$t = 0$')
#we could initialize next_event, but it's not necessary
#next_event = pc.CollisionEvent( Type = 'wall or other, to be determined', dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)

'''define parameters for MD simulation'''
total_time = 0.0
dt_frame = 0.02
NumberOfFrames = 1000
next_event = mols.compute_next_event()

#Here we take into account the time elapsed from first frame.

for j in range(len(mols.covid_status)):
    if mols.covid_status[j] == 1:
        mols.time_infected[j] += dt_frame
print("infected time", mols.time_infected)

def MolecularDynamicsLoop( frame ):
    '''
    The MD loop including update of frame for animation.
    '''
    global total_time, mols, next_event
    covid_case = NumberOfCovid
    
    #here we start a clock for each infected monomer, and add dt_frame to each clock.
    for j in range(len(mols.covid_status)):
        if mols.covid_status[j] == 1:
            covid_case += 1
            mols.time_infected[j] += dt_frame
            
 
    next_frame_time = total_time + dt_frame
    #Can't modify
    #Problem in this loop
    #next_event.dt < 0, problem !!!!!!
    while next_frame_time > total_time + next_event.dt:
        mols.pos += mols.vel * next_event.dt
        total_time += next_event.dt
        mols.compute_new_velocities( next_event )
        next_event = mols.compute_next_event()
        
            
    dt_remains = next_frame_time - total_time
    mols.pos += mols.vel * dt_remains
    total_time += dt_remains
    next_event.dt -= dt_remains

    #print( mols.__str__(0) ) #uncomment 
    
    # we can save additional snapshots for debugging -> slows down real-time animation
    #mols.snapshot( FileName = Snapshot_output_dir + '/Conf_t%.8f_0.png' % total_time, Title = '$t = %.8f$' % total_time)

    healthy = NumberOfMonomers - covid_case
    plt.suptitle( 'Covid Simulation with %.0f peoples' % (NumberOfMonomers), fontsize=20)
    plt.title('Covid cases = %.0f, Healthy = %.0f, \n $t = %.4f$, remaining frames = %d' % (covid_case, healthy, total_time, NumberOfFrames-(frame+1)), fontsize=10)
    plt.axis('off')
    collection.set_offsets( mols.pos )
    
    #updating the colors to give newly infected the corresponding color.
    MonomerColors = 2/3*mols.covid_status
    collection.set_array(MonomerColors)
    collection.set_clim(0, 1)
    
    return collection



'''We define and initalize the plot for the animation'''
fig, ax = plt.subplots()
L_xMin, L_yMin = mols.BoxLimMin #not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax #not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax.set_aspect('equal')

# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
ax.add_patch(rect)


# plotting all monomers as solid circles of individual color$
MonomerColors = 2/3*mols.covid_status #we want one color for infected and another for non infected.
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
collection.set_array(MonomerColors)
collection.set_clim(0, 1) # <--- we set the limit for the color code
ax.add_collection(collection)

'''Create the animation, i.e. looping NumberOfFrames over the update function'''
Delay_in_ms = 33.3 #dely between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames, interval=Delay_in_ms, blit=False, repeat=False)
plt.show()

'''Save the final configuration and make a snapshot.'''
#write the function to save the final configuration
#mols.save_configuration(Path_ToConfiguration)
mols.snapshot( FileName = Snapshot_output_dir + '/FinalConf.png', Title = '$t = %.4f$' % total_time)
