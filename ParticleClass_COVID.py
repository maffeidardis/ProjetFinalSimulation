import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle
import random
import time
            
class CollisionEvent:
    """
    Object contains all information about a collision event
    which are necessary to update the velocity after the collision.
    For MD of hard spheres (with hard bond-length dimer interactions)
    in a rectangular simulation box with hard walls, there are only
    two distinct collision types:
    1) wall collision of particle i with vertical or horizontal wall
    2) external (or dimer bond-length) collision between particle i and j
    """
    def __init__(self, Type = 'wall or other', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 1):
        """
        Type = 'wall' or other
        dt = remaining time until collision
        mono_1 = index of monomer
        mono_2 = if inter-particle collision, index of second monomer
        w_dir = if wall collision, direction of wall
        (   w_dir = 0 if wall in x direction, i.e. vertical walls
            w_dir = 1 if wall in y direction, i.e. horizontal walls   )
        """
        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only importent for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions
        
        
    def __str__(self):
        if self.Type == 'wall':
            return "Event type: {:s}, dt: {:.8f}, mono_1 = {:d}, w_dir = {:d}".format(self.Type, self.dt, self.mono_1, self.w_dir)
        else:
            return "Event type: {:s}, dt: {:.8f}, mono_1 = {:d}, mono_2 = {:d}".format(self.Type, self.dt, self.mono_1, self.mono_2)
class RecoveryEvent:
    '''
    Object contains all information about a recovery event which are necessary 
    to update the status of infection after a collision event.
    This class can be used to compute any covid related event only dependant
    of the elapsed time .
    '''
    def __init__(self, Type='recovery or other', dt = np.inf, mono = 0):
        self.Type = Type
        self.dt = dt
        self.mono = mono
    
    def __str__(self):
            if self.Type == 'recovery':
                return "Event type: {:s}, dt: {:.8f}, mono = {:d}".format(self.Type, self.dt, self.mono)
            else:
                return "Event type: {:s}, dt: {:.8f}, mono = {:d}".format(self.Type, self.dt, self.mono)
            
class Monomers:
    """
    Class for event-driven molecular dynamics simulation of hard spheres:
    -Object contains all information about a two-dimensional monomer system
    of hard spheres confined in a rectengular box of hard walls.
    -A configuration is fully described by the simulation box and
    the particles positions, velocities, radiai, and masses.
    -Initial configuration of $N$ monomers has random positions (without overlap)
    and velocities of random orientation and norms satisfying
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for inter-particle collsions is the mono_pair array,
    which book-keeps all combinations without repetition of particle-index
    pairs for inter-particles collisions, e.g. for $N = 3$ particles
    indices = 0, 1, 2
    mono_pair = [[0,1], [0,2], [1,2]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 7
    NumberMono_per_kind = [ 2, 5]
    Radiai_per_kind = [ 0.2, 0.5]
    Densities_per_kind = [ 2.2, 5.5]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2,...,mono_6 have radius 0.5 and mass 5.5*pi*0.5^2
    """
    def __init__(self, NumberOfMonomers = 4, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1):
        assert ( NumberOfMonomers > 0 )
        assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
        self.NM = NumberOfMonomers
        self.DIM = 2 #dimension of system
        self.BoxLimMin = np.array([ L_xMin, L_yMin])
        self.BoxLimMax = np.array([ L_xMax, L_yMax])
        self.mass = np.empty( self.NM ) # Masses, not initialized but desired shape
        self.rad = np.empty( self.NM ) # Radiai, not initialized but desired shape
        self.pos = np.empty( (self.NM, self.DIM) ) # Positions, not initalized but desired shape
        self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
        self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
        self.next_wall_coll = CollisionEvent( Type = 'wall', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 0)
        self.next_mono_coll = CollisionEvent( Type = 'mono', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 0)
    
        self.assignRadiaiMassesVelocities( NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
        self.assignRandomMonoPos( )

    
    def assignRadiaiMassesVelocities(self, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1 ):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        '''
        
        '''initialize radiai and masses'''
        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape))
        
        rad=[]
        densities=[]
        for number_mono in NumberMono_per_kind:
            for number in range(number_mono):
                rad_value = Radiai_per_kind[np.where(NumberMono_per_kind == number_mono)[0]]
                rad.append(rad_value[0])
                densities_value = Densities_per_kind[np.where(NumberMono_per_kind == number_mono)[0]]
                densities.append(densities_value[0])
        
        #mass is computed given the density of the particle and the area of the particle.
        self.rad = np.array(rad)
        self.mass = np.array(densities)*np.pi * self.rad**2

        
        '''initialize velocities'''
        assert( k_BT > 0 )

        #creation of a random list of orientation
        theta_list=np.random.uniform(0,2*np.pi,self.NM)
        
        #Initialising particle velocity with condition that total kinetic energy E_kin is constant.
        # E_kin = sum_i m_i /2 v_i^2 = N * dim/2 k_BT https://en.wikipedia.org/wiki/Ideal_gas_law#Energy_associated_with_a_gas
        #Initial configuration of $N$ monomers has velocities of random
        #orientation and norms satisfying
        #$E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
        #$k_B$ the Boltzmann constant, and $T$ the temperature.
        self.vel[:,0]=np.cos(theta_list)
        self.vel[:,1]=np.sin(theta_list)
        self.vel[:, 0] *= 2*k_BT/self.mass
        self.vel[:, 1] *= 2*k_BT/self.mass
    
    def assignRandomMonoPos(self, start_index = 0 ):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        Initialize random positions without overlap between monomers and wall.
        '''
        assert ( min(self.rad) > 0 )#otherwise not initialized
        mono_new, infiniteLoopTest = start_index, 0

        #Here we will define the maximum dimensions (considering the radium and also the dimensions of the box)
        #Here we use the np.newaxis to add one dimension.
        #So, we can calculate the extremal positions for each monomer.
        lim_dim_max = self.BoxLimMax - self.rad[:, np.newaxis]
        lim_dim_min = self.BoxLimMin + self.rad[:, np.newaxis]
        BoxLength = self.BoxLimMax - self.BoxLimMin
        
        while mono_new < self.NM and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1

            #Generating random positions based on the limits imposed.
            #The first is related to the x and the second one to the y.
            x_pos = np.random.uniform(lim_dim_min[mono_new][0], lim_dim_max[mono_new][0])
            y_pos = np.random.uniform(lim_dim_min[mono_new][1], lim_dim_max[mono_new][1])
            
            self.pos[mono_new] = [x_pos, y_pos]
            
            #Now, we have to deal with the overlap cases before moving to the next monomer:
            overlap = False

            mono_actual = 0
            while (mono_actual < mono_new) and not overlap:
                distance_mono_x = self.pos[mono_new, 0] - self.pos[mono_actual, 0]
                distance_mono_y = self.pos[mono_new, 1] - self.pos[mono_actual, 1]

                rad_sum = self.rad[mono_new] + self.rad[mono_actual]

                #Here we apply the radium condition to analyse if there is or there is not an overlap event:
                if distance_mono_y ** 2 + distance_mono_x ** 2 <= rad_sum ** 2:
                    overlap = True
                mono_actual += 1

            #If we don't have overlap, we can move forward to the next monomer:
            if not overlap:
                mono_new += 1
        #Just a condition to stop the program if it takes too much time to place monomers.
        if mono_new != self.NM:
            print('Failed to initialize all particle positions.\nIncrease simulation box size!')
            return(None)
        
    
    def __str__(self, index = 'all'):
        if index == 'all':
            return "\nMonomers with:\nposition = " + str(self.pos) + "\nvelocity = " + str(self.vel) + "\nradius = " + str(self.rad) + "\nmass = " + str(self.mass)
        else:
            return "\nMonomer at index = " + str(index) + " with:\nposition = " + str(self.pos[index]) + "\nvelocity = " + str(self.vel[index]) + "\nradius = " + str(self.rad[index]) + "\nmass = " + str(self.mass[index])
        
    def Wall_time(self):
        '''
        -Function computes list of remaining time dt until future
        wall collision in x and y direction for every particle.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_wall_coll.
        -Meaning of future:
        if v > 0: solve BoxLimMax - rad = x + v * dt
        else:     solve BoxLimMin + rad = x + v * dt
        '''
        
        x = self.pos[:, 0]
        y = self.pos[:, 1]

        vx = self.vel[:, 0]
        vy = self.vel[:, 1]
        
        #Computing  the time before next wall collision for each monomer:
        dtx = np.where(vx > 0, (self.BoxLimMax[0] - x - self.rad)/vx, (self.BoxLimMin[0] - x + self.rad)/vx)
        dty = np.where(vy > 0, (self.BoxLimMax[1] - y - self.rad)/vy, (self.BoxLimMin[1] - y + self.rad)/vy)
        index_min_x = np.argmin(dtx)
        index_min_y = np.argmin(dty)
        
        #Checking on wich side does the wall collision occur:
        if dtx[index_min_x] > dty[index_min_y]:
            minCollTime = dty[index_min_y]
            wall_direction = 1
            collision_disk = index_min_y

        else:
            minCollTime = dtx[index_min_x]
            wall_direction = 0
            collision_disk = index_min_x
            
        #Updating the time till collision for each monomer.
        self.next_wall_coll.dt = minCollTime
        self.next_wall_coll.mono_1 = collision_disk
        self.next_wall_coll.w_dir = wall_direction
        
        
    def Mono_pair_time(self):
        '''
        - Function computes list of remaining time dt until
        future external collition between all combinations of
        monomer pairs without repetition. Then, it stores
        collision parameters of the event with
        the smallest dt in the object next_mono_coll.
        - If particles move away from each other, i.e.
        scal >= 0 or Omega < 0, then remaining dt is infinity.
        '''
        mono_i = self.mono_pairs[:,0] # List of collision partner 1
        mono_j = self.mono_pairs[:,1] # List of collision partner 2

        #Here we will calculate all the variables that we will need to compute event
        delta_distances = self.pos[mono_i] - self.pos[mono_j]
        distances = np.linalg.norm(delta_distances, axis = 1)
        velocities = self.vel[mono_i] - self.vel[mono_j]

        #Here we calculate the a, b and c:
        a = np.linalg.norm(velocities, axis=1) ** 2
        b = 2*(delta_distances * velocities).sum(1)
        c = distances ** 2 - (self.rad[mono_i] + self.rad[mono_j]) ** 2

        dif_bac = b**2 - 4*a*c
        
        #Here we are trying to find the time with respect to our conditions and, if not, we change the value from the dt to np.inf
        dt_time = np.where(np.logical_and(b < 0, dif_bac > 0), (-b-np.sqrt(dif_bac))/(2*a), np.inf)
        tmin = np.argmin(dt_time)

        minCollTime = dt_time[tmin]
        collision_disk_1, collision_disk_2 = self.mono_pairs[tmin]
        
        #updating the time till next externel collision for each monomer:
        self.next_mono_coll.dt = minCollTime
        self.next_mono_coll.mono_1 = collision_disk_1
        self.next_mono_coll.mono_2 = collision_disk_2
        
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''

        self.Mono_pair_time()
        self.Wall_time()
        
        if self.next_wall_coll.dt < self.next_mono_coll.dt:
            return self.next_wall_coll
        else:
            return self.next_mono_coll

        
            
    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        '''
        
        #If the event is a wall collision, velocity is updated with opposite direction but same norm.
        if next_event.Type == 'wall':
            mono_index = next_event.mono_1
            wall_direction = next_event.w_dir
            self.vel[ mono_index , wall_direction ] *= -1
        
        #in the case of external collision there is a momentum transfer between particles.
        else:
            mono_1 = next_event.mono_1
            mono_2 = next_event.mono_2

            delta_pos = self.pos[mono_2] - self.pos[mono_1]
            delta_vel = self.vel[mono_1] - self.vel[mono_2]
            sum_mass = self.mass[mono_1] + self.mass[mono_2]
            delta_mass = self.vel[mono_1] - self.mass[mono_2]
            alpha = (2 /(sum_mass)) * np.inner(delta_pos,delta_vel)  * (delta_pos / np.linalg.norm(delta_pos)**2)
            self.vel[mono_1], self.vel[mono_2] = self.vel[mono_1] - self.mass[mono_2]* alpha, self.vel[mono_2] + self.mass[mono_1] * alpha

    def snapshot(self, FileName = './snapshot.png', Title = '$t = $?'):
        '''
        Function saves a snapshot of current configuration,
        i.e. particle positions as circles of corresponding radius,
        velocities as arrows on particles,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        MonomerColors = np.linspace( 0.2, 0.95, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig(FileName)
        plt.close()

class patient(Monomers):
    '''
    - Class derived from the monomer class.
    - Take into account the infection status of the particle and all the covid related attributes.
    '''
    def __init__(self, NumberOfMonomers = 4, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1, NumberOfCovid = 0 , base_probability_of_infection = 0, covid_duration = 0):    
        assert ( NumberOfMonomers > 0 )
        assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
        self.NM = NumberOfMonomers
        self.DIM = 2 #dimension of system
        self.BoxLimMin = np.array([ L_xMin, L_yMin])
        self.BoxLimMax = np.array([ L_xMax, L_yMax])
        self.mass = np.empty( self.NM ) # Masses, not initialized but desired shape
        self.rad = np.empty( self.NM ) # Radiai, not initialized but desired shape
        self.pos = np.empty( (self.NM, self.DIM) ) # Positions, not initalized but desired shape
        self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
        self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
        self.next_wall_coll = CollisionEvent( Type = 'wall', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 0)
        self.next_mono_coll = CollisionEvent( Type = 'mono', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 0)
    
        self.assignRadiaiMassesVelocities( NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
        self.assignRandomMonoPos( )
        
        #Here we define all the new attributes for COVID spread simulation.
        
        self.p_of_inf = np.zeros(self.NM) # probability of infecting for each monomer.
        self.covid_status = np.zeros(self.NM) # status of infection of each monomer.
        self.NC = NumberOfCovid # initial number of infected monomers.
        self.p_covid = base_probability_of_infection
        self.time_infected = np.zeros(self.NM)
        self.covid_duration = covid_duration
        self.next_recovery = RecoveryEvent(Type='recovery', dt = np.inf, mono = 0)
        self.assignRandomCovidMonomers()
        
    def assignRandomCovidMonomers(self):
        covid_index = np.random.randint(0, self.NM, self.NC)
        for j in range(0, len(covid_index)):
            self.covid_status[covid_index[j]] = 1
            self.p_of_inf[covid_index[j]] = self.p_covid
        self.NC = sum(self.covid_status)
            
    def Recovery_time(self):
         '''
         Function that compute the time remaining before a particle heals from COVID.
         '''
         if sum(self.covid_status) == 0:
             self.next_recovery.dt = np.inf
             self.next_recovery.mono = 0
             return None
         
         recovery_times = np.abs(self.time_infected - self.covid_duration)*self.covid_status
         recovery_times = np.where(recovery_times != 0, recovery_times, np.NAN)
         minRecoveryTime = np.nanmin(recovery_times)
         minRecoveryTime_index = np.nanargmin(recovery_times)
         self.next_recovery.dt = minRecoveryTime
         self.next_recovery.mono = minRecoveryTime_index
             
    def compute_next_event(self):
            '''
            Function gets event information about:
            1) next possible wall event
            2) next possible pair event
            Function returns event info of event with
            minimal time, i.e. the clostest in future.
            '''

            self.Mono_pair_time()
            self.Wall_time()
            self.Recovery_time()
            
            #Here we determine which one of the three event is the closest in time.
            if self.next_wall_coll.dt < self.next_mono_coll.dt and self.next_wall_coll.dt < self.next_recovery.dt:
                return self.next_wall_coll
            elif self.next_mono_coll.dt < self.next_recovery.dt:
                    return self.next_mono_coll
            else:
                return self.next_recovery
                
    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        *************
        This function now also update the covid status of a monomer.
        '''
        
        #If the event is a wall collision, velocity is updated with opposite direction but same norm.
        if next_event.Type == 'wall':
            mono_index = next_event.mono_1
            wall_direction = next_event.w_dir
            self.vel[ mono_index , wall_direction ] *= -1
        
        #in the case of external collision there is a momentum transfer between particles.
        elif next_event.Type == 'mono':
            mono_1 = next_event.mono_1
            mono_2 = next_event.mono_2

            delta_pos = self.pos[mono_2] - self.pos[mono_1]
            delta_vel = self.vel[mono_1] - self.vel[mono_2]
            sum_mass = self.mass[mono_1] + self.mass[mono_2]
            delta_mass = self.vel[mono_1] - self.mass[mono_2]
            alpha = (2 /(sum_mass)) * np.inner(delta_pos,delta_vel)  * (delta_pos / np.linalg.norm(delta_pos)**2)
            self.vel[mono_1], self.vel[mono_2] = self.vel[mono_1] - self.mass[mono_2]* alpha, self.vel[mono_2] + self.mass[mono_1] * alpha
            
            #Here we udpate the infected status of the monomers if one of them is infected and not the other.
            
            covid_status_pair = self.covid_status[mono_1], self.covid_status[mono_2] #we recover the status of infection of the two colliding particles.
            if sum(covid_status_pair) == 1:
                infected_pair= [mono_1, mono_2]
                next_infected = infected_pair[covid_status_pair.index(0)]
                infector = infected_pair[covid_status_pair.index(1)]
                #Here we will considerate it to try to work with probability with vaccines for example
                p_infected = self.p_of_inf[next_infected] + self.p_of_inf[next_infected] #probability for the non infected to get COVID.
                p_infector = self.p_of_inf[infector] #orobability for the infected to give COVID.
                p_of_transmission = p_infected*p_infector
                trial = random.random()
                if trial <= p_infector:
                    self.covid_status[next_infected] = 1
                    self.p_of_inf[next_infected] = self.p_covid
                    
            #Here we deal with recovery events, ie we reverse the infection status.
        else: 
            mono = self.next_recovery.mono
            self.covid_status[mono] = 1/4
            self.time_infected[mono] = 0 
            self.p_of_inf[mono] = 1 - self.p_covid
        
class Dimers(Monomers):
    """
    --> Class derived from Monomers.
    --> See also comments in Monomer class.
    --> Class for event-driven molecular dynamics simulation of hard-sphere
    system with DIMERS (and monomers). Two hard-sphere monomers form a dimer,
    and experience additional ellastic collisions at the maximum
    bond length of the dimer. The bond length is defined in units of the
    minimal distance of the monomers, i.e. the sum of their radiai.
    -Next to the monomer information, the maximum dimer bond length is needed
    to fully describe one configuration.
    -Initial configuration of $N$ monomers has random positions without overlap
    and separation of dimer pairs is smaller than the bond length.
    Velocities have random orientations and norms that satisfy
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for all inter-particle collsions is the mono_pair array
    (explained in momonmer class). Essentail for the ellastic bond collision
    of the dimers is the dimer_pair array which book-keeps index pairs of
    monomers that form a dimer. For example, for a system of $N = 10$ monomers
    and $M = 2$ dimers:
    monomer indices = 0, 1, 2, 3, ..., 9
    dimer_pair = [[0,2], [1,3]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 10
    NumberOfDimers = 2
    bond_length_scale = 1.2
    NumberMono_per_kind = [ 2, 2, 6]
    Radiai_per_kind = [ 0.2, 0.5, 0.1]
    Densities_per_kind = [ 2.2, 5.5, 1.1]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2, mono_3 have radius 0.5 and mass 5.5*pi*0.5^2
    and monomers mono_4,..., mono_9 have radius 0.1 and mass 1.1*pi*0.1^2
    dimer pairs are: (mono_0, mono_2), (mono_1, mono_3) with bond length 1.2*(0.2+0.5)
    see bond_length_scale and radiai
    """
    def __init__(self, NumberOfMonomers = 4, NumberOfDimers = 2, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), bond_length_scale = 1.2, k_BT = 1):
        #if __init__() defined in derived class -> child does NOT inherit parent's __init__()
        assert ( (NumberOfDimers > 0) and (NumberOfMonomers >= 2*NumberOfDimers) )
        assert ( bond_length_scale > 1. ) # is in units of minimal distance of respective monomer pair
        Monomers.__init__(self, NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
        self.ND = NumberOfDimers
        self.dimer_pairs = np.array([[k,self.ND+k] for k in range(self.ND)])#choice 2 -> more practical than [2*k,2*k+1]
        mono_i, mono_j = self.dimer_pairs[:,0], self.dimer_pairs[:,1]
        self.bond_length = bond_length_scale * ( self.rad[mono_i] + self.rad[mono_j] )
        self.next_dimer_coll = CollisionEvent( Type = 'dimer', dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)
        
        '''
        Positions initialized as pure monomer system by monomer __init__.
        ---> Reinitalize all monomer positions, but place dimer pairs first
        while respecting the maximal distance given by the bond length!
        '''
        self.assignRandomDimerPos()
        self.assignRandomMonoPos( 2*NumberOfDimers )
    
  
