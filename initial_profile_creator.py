import numpy as np
from scipy.stats import norm

class EqualWeightParticleDistribution:
    def __init__(self,
                 simulation_obj,
                 rms_velocity: list | np.ndarray,
                 density_profile_z: np.ndarray,
                 density_profile: np.ndarray,
                 mean_velocity: list | np.ndarray = [0.0, 0.0, 0.0]
                 ):
        '''
        Class to create a particle distribution with equal weights and
        uniformly spaced positions of a particle distribution in in 1D3V
        phase space.

        Parameters
        ----------
        rms_velocity: list | np.ndarray
            RMS velocity in each direction, [vx, vy, vz]
        density_profile_z: np.ndarray
            Positions of the density profile values
        density_profile: np.ndarray
            Density profile
        mean_velocity: float, default = [0.0, 0.0, 0.0]
            Mean velocity in each direction, [vx, vy, vz]
        '''
        self.nppc = simulation_obj.seed_nppc
        self.nz = simulation_obj.nz
        self.dz = simulation_obj.dz

        self.zmin = simulation_obj.zmin
        self.zmax = simulation_obj.zmax

        if len(rms_velocity) != 3:
            raise ValueError('RMS velocity should be a list of length 3')
        if len(mean_velocity) != 3:
            raise ValueError('Mean velocity should be a list of length 3')

        if any([i < 0 for i in rms_velocity]):
            raise ValueError('RMS velocity should be positive')

        self.get_number_of_particles(density_profile, density_profile_z)
        self.get_particle_weight()
        self.distribute_particles(rms_velocity, mean_velocity)

    def get_number_of_particles(self, density_profile: np.ndarray, density_profile_z: np.ndarray):
        '''
        Determine the number of particles in the distribution
        '''
        cell_centers = np.linspace(self.dz / 2, self.zmax - self.dz / 2, self.nz)
        self.density_profile = np.interp(cell_centers, density_profile_z, density_profile)

        normalized_density_profile = self.density_profile / np.max(self.density_profile)
        self.nppc_profile = np.round(self.nppc * normalized_density_profile).astype(int)

        self.total_particles = int(np.sum(self.nppc_profile))

    def get_particle_weight(self):
        '''
        Get the (uniform) weight of the particles
        '''
        w = np.max(self.density_profile) / self.nppc * self.dz
        self.weight = np.ones(self.total_particles) * w

    def distribute_particles(
                             self,
                             rms_velocity: list | np.ndarray,
                             mean_velocity: list | np.ndarray
                             ):
        '''
        Distribute particles in the domain
        '''
        self.x = np.zeros(self.total_particles)
        self.y = np.zeros(self.total_particles)
        self.z = np.zeros(self.total_particles)
        self.ux = np.zeros(self.total_particles)
        self.uy = np.zeros(self.total_particles)
        self.uz = np.zeros(self.total_particles)

        self.get_z_positions()
        self.get_velocities(rms_velocity, mean_velocity)

    def get_z_positions(self):
        '''
        Place particles uniformly randomly across each cell
        '''
        start = 0
        for i in range(self.nz):
            end = start + self.nppc_profile[i]
            self.z[start:end] = np.random.uniform(
                self.zmin + i * self.dz,
                self.zmin + (i + 1) * self.dz,
                end - start)
            start = end

    def get_velocities(self, rms_velocity: list | np.ndarray, mean_velocity: list | np.ndarray):
        '''
        Get velocities for the particles from Maxwellian distribution
        with mean velocity and RMS velocity spread
        '''
        self.vrms = np.array(rms_velocity)
        self.vz0 = np.array(mean_velocity)

        self.ux = norm.rvs(loc=self.vz0[0], scale=self.vrms[0], size=self.total_particles)
        self.uy = norm.rvs(loc=self.vz0[1], scale=self.vrms[1], size=self.total_particles)
        self.uz = norm.rvs(loc=self.vz0[2], scale=self.vrms[2], size=self.total_particles)