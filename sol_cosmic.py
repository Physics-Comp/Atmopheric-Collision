import numpy as np

# Important constants
particle_names = ['proton', 'electron', 'pion', 'muon', 'neutrino', 'N14']

# All masses in MeV/c^2
particle_masses = {
    'proton': 938.3,
    'pion': 139.6,
    'electron': 0.5110,
    'muon': 105.7,
    'neutrino': 0.,
    'N14': 13040.0,
}

tau_p_0 = 1.14e-5  # s, at sea level
tau_pi = 2.603e-8 # s
tau_mu = 2.197e-6 # s

b = 1 / (500)  # 1/MeV

def particle_mass(name):
    if not name in particle_masses.keys():
        raise ValueError("No mass defined for particle {}!".format(name))
        
    return particle_masses[name]

def lorentz_beta_gamma(beta_x, beta_y):
    """
    Computes beta and gamma from the components
    of speed (beta_x, beta_y).
    """
    
    beta = np.sqrt(beta_x**2 + beta_y**2)

    if (beta >= 1.):
        print(beta_x, beta_y, beta)
        raise ValueError("Should never try to boost with beta >= 1!")
    
    gamma = 1 / np.sqrt(1 - beta**2)
    
    return (beta, gamma)
    
def boost_momentum(m, px, py, beta_x, beta_y):
    # Note: need to clarify sign of betas here...(?)
    E = np.sqrt(px**2 + py**2 + m**2)
    
    beta, gamma = lorentz_beta_gamma(beta_x, beta_y)
    
    px_lab = px
    px_lab -= gamma * beta_x * E
    px_lab += (gamma - 1) * beta_x * (beta_x * px + beta_y * py) / beta**2
    
    py_lab = py
    py_lab -= gamma * beta_y * E
    py_lab += (gamma - 1) * beta_y * (beta_x * px + beta_y * py) / beta**2
    
    return (px_lab, py_lab)
    
def momentum_to_velocity(m, px, py):
    # Massless case is special, preserve direction and v=c=1
    if m == 0:
        p = np.sqrt(px**2 + py**2)
        
        return ( px / p,  py / p)
    
    gamma = np.sqrt(1 + (px**2 + py**2) / m**2)
    
    beta_x = (px / m) / gamma
    beta_y = (py / m) / gamma
    
    return (beta_x, beta_y)

def velocity_to_momentum(m, beta_x, beta_y):
    # Massless case is an error here, no way to get momentum from velocity if m=0!
    if m == 0:
        raise ValueError("velocity_to_momentum is not defined for massless particles!")
    
    (beta, gamma) = lorentz_beta_gamma(beta_x, beta_y)
    
    px = gamma * m * beta_x
    py = gamma * m * beta_y
    
    return (px, py)

        
def check_for_interaction(particle, dt, h):
    """
    Test to see if a particle will decay/scatter
    within timestep dt.
    
    Arguments:
    ===
    - particle_name: Name of particle to test.
    - beta_x, beta_y: relativistic speeds (v/c) in x and y directions.
    - dt: time interval (as a Unyt object, in seconds)
    
    Returns:
    ===
    True if the particle decay/scatters, False otherwise.
    """
    
    # Unpack particle
    (particle_name, p_x, p_y, traj_x, traj_y) = particle
    
    assert particle_name in particle_names
    
    if particle_name in ['electron', 'neutrino']:
        # These two are stable in our model
        return False

    (beta_x, beta_y) = momentum_to_velocity(particle_mass(particle_name), p_x, p_y)
    try:
        (beta, gamma) = lorentz_beta_gamma(beta_x, beta_y)
    except:
        print(particle)
        raise
        
    if particle_name == 'proton':
        # Must be above threshold energy
        Ep = np.sqrt(p_x**2 + p_y**2 + particle_mass('proton')**2)
        if Ep <= 1220:
            return False
        
        # Interaction time depends on height
        # exp(-h/7 km) in light seconds --> exp(-h / (2.33 x 10^{-5} ls))
        tau = tau_p_0 / beta * np.exp(-h / (2.33e-5))
    elif particle_name == 'pion':
        tau = gamma * tau_pi
    elif particle_name == 'muon':
        tau = gamma * tau_mu

    p = 1 - np.exp(-(dt / tau))

    return np.random.rand() < p
    

def particle_decay(name):
    if name == 'pion':
        M_X = particle_mass('pion')
        M_Y = particle_mass('muon')
        name_Y = 'muon'
    elif name == 'muon':
        M_X = particle_mass('muon')
        M_Y = particle_mass('electron')
        name_Y = 'electron'
    else:
        print("Should never get here!")
        assert False
        
    # Choose a random decay angle
    theta = np.random.rand() * 2 * np.pi
    
    # Find total momentum
    p = (M_X**2 - M_Y**2) / (2 * M_X)
    
    # Return decay product momentum tuples
    product_Y = (name_Y, M_Y, p * np.cos(theta), p * np.sin(theta) )
    product_Z = ('neutrino', 0.0, -p * np.cos(theta), -p * np.sin(theta) )
    
    return [product_Y, product_Z]    

def proton_downscatter(beta_x, beta_y):
    # Nitrogen nucleus is at rest in lab frame
    # Boost into the proton rest frame
    m_N14 = particle_mass('N14')
    p_N_x, p_N_y = boost_momentum(m_N14, 0.0, 0.0, beta_x, beta_y)
    
    
    # Two random angles for the two pions
    theta_1 = np.random.rand() * 2 * np.pi
    theta_2 = np.random.rand() * 2 * np.pi
    
    m_pi = particle_mass('pion')
    
    # Random energy for the pions
    E_pi_1 = m_pi - (1/b) * np.log(1 - np.random.rand())
    E_pi_2 = m_pi - (1/b) * np.log(1 - np.random.rand())
    
    # Pion momentum from energy
    p_pi_1 = np.sqrt(E_pi_1**2 - m_pi**2)
    p_pi_2 = np.sqrt(E_pi_2**2 - m_pi**2)
    
    # Build decay product momentum tuples
    product_pi_1 = ('pion', m_pi, p_pi_1 * np.cos(theta_1), p_pi_2 * np.sin(theta_1))
    product_pi_2 = ('pion', m_pi, p_pi_2 * np.cos(theta_2), p_pi_2 * np.sin(theta_2))
    
    # Calculate new energy of the nucleus, ignoring proton motion
    E_N = np.sqrt(p_N_x**2 + p_N_y**2 + m_N14**2)
    p_N = np.sqrt(p_N_x**2 + p_N_y**2)
    E_N_prime = E_N - E_pi_1 - E_pi_2
    p_N_prime = np.sqrt(E_N_prime**2 - m_N14**2)
    
    delta_p_N_x, delta_p_N_y = p_N_x * (1 - p_N_prime / p_N), p_N_y * (1 - p_N_prime / p_N)
        
    # Momentum conservation: proton
    product_p = ('proton', particle_mass('proton'), delta_p_N_x -product_pi_1[2] - product_pi_2[2], delta_p_N_y -product_pi_1[3] - product_pi_2[3])

    return [product_pi_1, product_pi_2, product_p]

def boost_product_to_lab(product, beta_x, beta_y):
    """
    Boost decay product momenta back to the lab frame.
    Note the boost is *negative* in the provided betas, since we're boosting back 
    from a moving frame.
    """
    # Unpack product
    (product_name, product_m, product_px, product_py) = product

    # Find momenta in lab frame
    (px_lab, py_lab) = boost_momentum(product_m, product_px, product_py, -beta_x, -beta_y)

    # Return
    return (product_name, px_lab, py_lab)


    
def run_cosmic_MC(particles, dt, Nsteps):
    """
    Main loop for cosmic ray Monte Carlo project.
    
    Arguments:
    =====
    * particles: a list of any length, containing "particle" tuples.
        A particle tuple has the form:
            (name, p_x, p_y, traj_x, traj_y)
        where p_x and p_y are momentum vector components, and
        traj_x and traj_y are NumPy arrays of distance.
        
    * dt: time between Monte Carlo steps, in seconds.
    * Nsteps: Number of steps to run the Monte Carlo before returning.
    
    Returns:
    =====
    A list of particle tuples.
    
    Example usage:
    =====
    
    >> init_p_x, init_p_y = velocity_to_momentum(particle_mass('muon'), 0.85, -0.24) # beta ~ 0.8, relativistic
    >> init_particles = [ ('muon', init_p_x, init_p_y, np.array([0]), np.array([1e-4]) ) ]  # ~ 30 km height in light-sec
    >> particles = run_cosmic_MC(init_particles, 1e-5, 100)
    
    The 'particles' variable after running should contain three particle tuples: the
    initial muon, an electron, and a neutrino.  (Even though the muon decays,
    we keep it in the final particle list for its trajectory.)
    
    """
    
    stopped_particles = []
    for step_i in range(Nsteps):
        updated_particles = []
        
        for particle in particles:
            # Unpack particle tuple
            (name, p_x, p_y, traj_x, traj_y) = particle
            (beta_x, beta_y) = momentum_to_velocity(particle_mass(name), p_x, p_y)

            # Check for interaction
            try:
                does_interact = check_for_interaction(particle, dt, traj_y[-1])
            except:
                print(particles[-5:])
                raise
            
            if does_interact:                
                if name == 'proton':
                    decay_products = proton_downscatter(beta_x, beta_y)
                else:
                    stopped_particles.append(particle)
                    decay_products = particle_decay(name)
                    
                # Transform products back to lab frame
                for product in decay_products:
                    (product_name, product_p_x, product_p_y) = boost_product_to_lab(product, beta_x, beta_y)
                    
                    # If this was a proton scatter, then the "new" proton is
                    # the same as the original, so keep track of its trajectory!
                    if name == 'proton' and product_name == 'proton':
                        product_traj_x = traj_x
                        product_traj_y = traj_y
                    else:
                        product_traj_x = np.array([traj_x[-1]])
                        product_traj_y = np.array([traj_y[-1]])
                
                    # Make new particle tuple and append
                    product_particle = (product_name, product_p_x, product_p_y, 
                                        product_traj_x, product_traj_y)
                    updated_particles.append( product_particle )
            else:
                # Doesn't interact, so compute motion
                traj_x = np.append(traj_x, traj_x[-1] + beta_x * dt)
                traj_y = np.append(traj_y, traj_y[-1] + beta_y * dt)

                updated_particles.append( (name, p_x, p_y, traj_x, traj_y) )
                
        
        # Run next timestep
        particles = updated_particles
    
    # Add stopped particles back to list and return
    particles.extend(stopped_particles)
    return particles
    
    
    
    
        
    