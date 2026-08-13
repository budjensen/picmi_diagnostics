from __future__ import annotations  # Allows using class names as type hints before they are fully defined
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inputs import CapacitiveDischargeExample  # Only imports Simulation for type checking to avoid circular import

import numpy as np
import sys, os

from pywarpx import fields, particle_containers, picmi, collision_trackers, power_deposition_trackers
from mpi4py import MPI as mpi
import time

# Initialize mpi communicator
comm = mpi.COMM_WORLD
num_proc = comm.Get_size()

constants = picmi.constants

class SEE:
    def __init__(self,
                 simulation_obj: CapacitiveDischargeExample,
                 sim_ext: picmi.Simulation.extension,
                 SEE_probability: float,
                 SEE_energy: float,
                 SEE_spec_names: list = None,
                 electron_species_name: str = 'electrons'
                 ):
        '''
        Class to calculate secondary electron emission (SEE) in a plasma.

        If SEE_probability > 1, make sure to install the callback function
        SEE.do_gamma_gt_1_SEE() in the WarpX picmi script.

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        sim_ext: picmi.Simulation.extension
            Simulation extension object
        SEE_probability: float
            SEE probability
        SEE_energy: float
            Energy of secondary electrons, in eV
        SEE_spec_names: list, optional
            List of species names that can yield SEE
        electron_species_name: str, optional
            Name of the electron species in the simulation. Default is 'electrons'.
        '''
        # Import simulation extension object
        self.sim_ext = sim_ext

        # Import simulation parameters
        self.zmax = simulation_obj.zmax
        self.dt   = simulation_obj.dt

        # Save the SEE probability and species
        self.SEE_probability = SEE_probability
        self.SEE_spec_names  = SEE_spec_names
        self.SEE_velocity    = np.sqrt(2 * SEE_energy * constants.q_e / constants.m_e)
        self.SEE_energy_J    = SEE_energy * constants.q_e  # Kinetic energy of each emitted secondary electron, in Joules
        self.electron_species_name = electron_species_name

        if self.SEE_probability < 0:
            raise ValueError('SEE_probability ERROR: SEE probability must be greater than or equal to 0.')

        # Automatically rescale the SEE probability if > 1
        self.min_SEE_to_inject = 0
        if self.SEE_probability > 1:
            self.min_SEE_to_inject = int(self.SEE_probability)
            self.SEE_probability -= self.min_SEE_to_inject

    def do_SEE(self):
        '''
        Function to calculate secondary electron emission.
        '''
        # Get wrappers
        buffer = particle_containers.ParticleBoundaryBufferWrapper()
        elec_pc = particle_containers.ParticleContainerWrapper(self.electron_species_name)
        lev = 0  # level 0 (no mesh refinement here)

        self.SEE_current_this_step = {
            'z_lo': 0,
            'z_hi': 0
        }
        for boundary in ['z_lo', 'z_hi']:

            # Initialize z
            if boundary == 'z_lo':
                z = 0.0
                root = 0
            else:
                z = self.zmax
                root = num_proc - 1

            for species in self.SEE_spec_names:
                try:
                    w = np.concatenate(buffer.get_particle_scraped_this_step(species, boundary, "w", lev))
                    delta_t = np.concatenate(buffer.get_particle_scraped_this_step(species, boundary, "deltaTimeScraped", lev))
                except ValueError:
                    w = np.array([])
                    delta_t = np.array([])

                if len(w) == 0:
                    if comm.rank == root:
                        self.send_SEE_number_and_weight(0)
                        continue
                    else:
                        num_SEE_to_add, weights = self.receive_SEE_number_and_weight(root)

                        if num_SEE_to_add == 0:
                            continue

                        self.SEE_current_this_step[boundary] += np.sum(weights)

                        # Call elec_pc.add_particles() to prevent a hang
                        elec_pc.add_particles()

                else:
                    # Determine if SEE occurs for each particle
                    SEE_occurs = np.random.uniform(size=len(w)) <= self.SEE_probability
                    n_SEE = np.sum(SEE_occurs)

                    we = w[SEE_occurs] # account for variable weights
                    delta_te = delta_t[SEE_occurs]

                    self.send_SEE_number_and_weight(n_SEE, weights=we)

                    if n_SEE == 0:
                        continue

                    # Get the velocity angles isotropically distribued on a unit hemisphere
                    phi = 2 * np.pi * np.random.uniform(size=n_SEE)
                    costheta = np.random.uniform(size=n_SEE)
                    sintheta = np.sqrt(1 - costheta**2)
                    vsintheta = self.SEE_velocity * sintheta
                    ux = vsintheta * np.cos(phi)
                    uy = vsintheta * np.sin(phi)
                    if boundary == 'z_lo':
                        uz = self.SEE_velocity * costheta
                    else:
                        uz = -self.SEE_velocity * costheta

                    self.SEE_current_this_step[boundary] += np.sum(we)

                    elec_pc.add_particles(
                        z=z + (self.dt - delta_te) * uz,
                        ux=ux,
                        uy=uy,
                        uz=uz,
                        w=we,
                        unique_particles=True,
                    )
                    # Note: Doing unique_particles=True will not add the same particle to each process
                    # if we only call elec_pc.add_particles() one on process. Since we already know which
                    # rank the particles should be added to (since it should be either end of the simulation)
                    # we can cheese the system and call elec_pc.add_particles() on all processes (so the simulation
                    # doesn't hang) and then only add the particles on the correct rank. This is a bit of a hack but
                    # works and is probably even faster than adding them since we'd need to send the info out and back

    def do_gamma_gt_1_SEE(self):
        '''
        Function to calculate secondary electron emission if SEE_probability > 1.
        '''
        # Get wrappers
        buffer = particle_containers.ParticleBoundaryBufferWrapper()
        elec_pc = particle_containers.ParticleContainerWrapper(self.electron_species_name)
        lev = 0  # level 0 (no mesh refinement here)

        self.SEE_current_this_step = {
            'z_lo': 0,
            'z_hi': 0
        }
        for boundary in ['z_lo', 'z_hi']:

            # Initialize z
            if boundary == 'z_lo':
                z = 0.0
                root = 0
            else:
                z = self.zmax
                root = num_proc - 1

            for species in self.SEE_spec_names:
                try:
                    w = np.concatenate(buffer.get_particle_scraped_this_step(species, boundary, "w", lev))
                    delta_t = np.concatenate(buffer.get_particle_scraped_this_step(species, boundary, "deltaTimeScraped", lev))
                except ValueError:
                    w = np.array([])
                    delta_t = np.array([])

                if len(w) == 0:
                    if comm.rank == root:
                        self.send_SEE_number_and_weight(0)
                        continue
                    else:
                        num_SEE_to_add, weights = self.receive_SEE_number_and_weight(root)

                        if num_SEE_to_add == 0:
                            continue

                        self.SEE_current_this_step[boundary] += np.sum(weights)

                        elec_pc.add_particles()

                else:
                    # Calculate the number of guaranteed secondary electrons to add
                    n_SEE = self.min_SEE_to_inject * len(w)

                    # Populate with the correct values of w and delta_t
                    w_prefix = np.repeat(w, self.min_SEE_to_inject)
                    delta_t_prefix = np.repeat(delta_t, self.min_SEE_to_inject)

                    # Determine if SEE occurs for each particle
                    SEE_occurs = np.random.uniform(size=len(w)) <= self.SEE_probability
                    n_SEE += np.sum(SEE_occurs) # uncomment for SEE_probability > 1

                    we = w[SEE_occurs] # account for variable weights
                    delta_te = delta_t[SEE_occurs]

                    # Prepend the guaranteed secondary electrons
                    we = np.concatenate((w_prefix, we))
                    delta_te = np.concatenate((delta_t_prefix, delta_te))

                    self.send_SEE_number_and_weight(n_SEE, weights=we)

                    if n_SEE == 0:
                        continue

                    # Get the velocity angles isotropically distribued on a unit hemisphere
                    phi = 2 * np.pi * np.random.uniform(size=n_SEE)
                    costheta = np.random.uniform(size=n_SEE)
                    sintheta = np.sqrt(1 - costheta**2)
                    vsintheta = self.SEE_velocity * sintheta
                    ux = vsintheta * np.cos(phi)
                    uy = vsintheta * np.sin(phi)
                    if boundary == 'z_lo':
                        uz = self.SEE_velocity * costheta
                    else:
                        uz = -self.SEE_velocity * costheta

                    self.SEE_current_this_step[boundary] += np.sum(we)

                    elec_pc.add_particles(
                        z=z + (self.dt - delta_te) * uz,
                        ux=ux,
                        uy=uy,
                        uz=uz,
                        w=we,
                        unique_particles=True,
                    )
                    # Note: Doing unique_particles=True will not add the same particle to each process
                    # if we only call elec_pc.add_particles() one on process. Since we already know which
                    # rank the particles should be added to (since it should be either end of the simulation)
                    # we can cheese the system and call elec_pc.add_particles() on all processes (so the simulation
                    # doesn't hang) and then only add the particles on the correct rank. This is a bit of a hack but
                    # works and is probably even faster than adding them since we'd need to send the info out and back

    def send_SEE_number_and_weight(self,
                                   num_SEE_to_add: int,
                                   weights: np.ndarray = None
                                   ) :
        '''
        Send the number of secondary electrons and their weights to all processes.

        Parameters
        ----------
        num_SEE_to_add: str
            number of secondary electrons to add
        weights: float, optional
            weight of the secondary electrons to add
        '''
        # Send the number of secondary electrons to add
        comm.Bcast(np.array([num_SEE_to_add], dtype='i'), root=comm.rank)

        # Only send weights if we're adding SEE particles and weights is provided
        if num_SEE_to_add > 0 and weights is not None:
            comm.Bcast(np.array(weights, dtype='d'), root=comm.rank)

    def receive_SEE_number_and_weight(self,
                                      root: int,
                                      ) -> tuple[int, np.ndarray]:
        '''
        Receive the number of secondary electrons to add from the root process.

        Parameters
        ----------
        root: int
            Root process to receive from

        Returns
        -------
        tuple
            (num_SEE_to_add, weights) where weights is the array of particle weights
        '''
        # Get the number of secondary electrons to add
        num_SEE_to_add = np.array([0], dtype='i')
        comm.Bcast(num_SEE_to_add, root=root)

        # Only receive weights if we're adding SEE particles
        if num_SEE_to_add[0] > 0:
            weights = np.array([0] * num_SEE_to_add[0], dtype='d')
            comm.Bcast(weights, root=root)
            return num_SEE_to_add[0], weights  # Return the entire weights array
        else:
            return 0, np.array([], dtype='d')

    def concat(list_of_arrays):
        if len(list_of_arrays) == 0:
            # Return a 1d array of size 0
            return np.empty(0)
        else:
            return np.concatenate(list_of_arrays)

class Diagnostics1D:

    PARTICLE_DIAGNOSTIC_PREFIXES = ['N', 'W', 'Wx', 'Wy', 'Wz', 'Jz', 'Jy', 'Jx', 'P_C', 'P_I', 'Pw', 'EDF', 'ExDF', 'EyDF', 'EzDF']
    FIELD_DIAGNOSTICS = ['E_z', 'E_y', 'E_x', 'phi', 'J_d', 'J_w']

    def __init__(self,
                 simulation_obj: CapacitiveDischargeExample,
                 sim_ext: picmi.Simulation.extension,
                 controls: dict,
                 SEE_obj: SEE = None,
                 interval_times: list = None,
                 diag_outfolder: str = './diags',
                 restart_checkpoint: bool = False,
                 interval_tolerance: float = 0.0
                ):
        '''
        Class to perform diagnostics in 1D WarpX simulations. Make sure
        to install all native WarpX diagnostics and checkpoints and do
        initialize_inputs() and initialize_warpx() before initializing
        this class.

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        sim_ext: picmi.Simulation.extension
            Simulation extension object
        SEE_obj: SEE, optional
            An object of the SEE class, if SEE is turned on
        species_controls: dict, optional
            Dictionary of diagnostic control switches and species info.
            One of the species must be electrons, or else the EDF diagnostics
            will not correctly bin energies. If not using electrons, you will need to fix
            this.
        interval_times: list, optional
            List of times to perform interval diagnostics, values must fall
            within the range [0, 1)
        diag_outfolder: str, optional
            Folder to save diagnostics
        restart_checkpoint: bool, optional
            Whether simulation is restarting from a checkpoint
        interval_tolerance: float, optional
            Tolerance factor for interval diagnostics collection. A value of 0.1
            means collect diagnostics within ±5% of the interval period around
            the exact collection time.

        Notes
        -----
        - The control dictionary needs to be formatted like, where any
        omitted keys will be turned off by default:

        species_controls = {
            'particle': {
                'species_name': {
                    'time_averaged': {
                        'N': True,
                        'W': True,
                        'Jz': True,
                        'P_C': True,  # Must have enable_power_deposition_tracking=True on the species
                        'P_I': True,  # Must have enable_power_deposition_tracking=True on the species
                        'Pw': True, # Power deposited to the walls by this species [W/m^2]
                        'EDF': True,
                        'ExDF': True,
                        'EyDF': True,
                        'EzDF': True,
                    },
                    'time_resolved': {
                        'N': True,
                    },
                    'interval': {
                        'Jz': True,
                        'P_I': True,
                    },
                    'properties': {
                        'Z': -1,
                        'm': m_species,
                        'max_edf': edf_max_eV,
                        'num_bins_edf': num_bins,
                        'max_exdf': edf_max_eV,
                        'max_eydf': edf_max_eV,
                        'max_ezdf': edf_max_eV,
                        'num_bins_exdf': 2 * num_bins,
                        'num_bins_eydf': 2 * num_bins,
                        'num_bins_ezdf': 2 * num_bins,
                    }
                },
            },
            'field': {
                'time_averaged': {
                    'E_z': True,
                    'E_y': False, # This should be zero unless running an ICP simulation
                    'E_x': False, # This should be zero unless you explicitly set it with wrappers
                    'phi': True,
                    'J_d': True,
                    'J_w': True
                },
            },
            'collision': {
                'coll_name_1': True, # WarpX collision name. Must have enable_collision_tracking=True
                                     # in the MCCCollisions or RecombinationCollisions object.
                                     # Saves collision rate [m^-3 s^-1] and energy transfer rate
                                     # [W/m^3] for each scattering process to the time_averaged folder.
                'coll_name_2': False,
            },
            'ieadfs': {'z_lo': True, 'z_hi': True},
            'eeadfs': {'z_lo': False, 'z_hi': False},
            'time_resolved_power': {'Pin_vst': False, ...},
        }

        - Interval times (if turned on) need to be formatted like:

        interval_times = [time1, time2, ...]

        }
        '''
        if controls is None:
            error_msg = 'Input ERROR: species_controls must be provided to specify diagnostics to collect.\n'
            raise ValueError(error_msg)

        # Parse species_controls
        # Extract species info and build flat dicts
        ieadfs = controls.get('ieadfs', {'z_lo': False, 'z_hi': False})
        eeadfs = controls.get('eeadfs', {'z_lo': False, 'z_hi': False})
        self.tr_power_dict = controls.get('time_resolved_power', {'Pin_vst': False})
        time_averaged_dict, time_resolved_dict, interval_dict, collisional_dict = self._parse_species_controls_dict(controls)

        # Correct any power dictionary values
        if self.tr_power_dict['Pin_vst']:
            for species in self.species_names:
                time_resolved_dict[f'Jz_{species}'] = True
            time_resolved_dict['phi'] = True

        # Import simulation parameters
        self.m_ion = simulation_obj.m_ion
        self.rf_period = 1 / simulation_obj.freq
        self.in_period = simulation_obj.interval_time
        self.dt = simulation_obj.dt
        self.nz = simulation_obj.nz
        self.dz = simulation_obj.dz
        self.nodes = np.linspace(simulation_obj.zmin, simulation_obj.zmax, self.nz + 1)
        self.interval_tolerance = interval_tolerance

        # Set simulation extension object
        self.sim_ext = sim_ext

        # Set external class objects
        self.SEE_obj = SEE_obj

        # General diagnostics are collected in three types:
        #  1. Time averaged
        #  2. Time resolved
        #  3. Interval sliced

        self.restart_checkpoint = restart_checkpoint

        if interval_times is None:
            self.in_slices = np.array([0, 0.125, 0.25, 0.375 , 0.5]) # Range [0, 1), fractions of interval_period
        else:
            self.in_slices = np.array(interval_times)

            # Check that times fall in the range [0, 1)
            self.in_slices = self.in_slices[self.in_slices < 1]
            self.in_slices = self.in_slices[self.in_slices >= 0]

            # Order times
            self.in_slices = np.sort(self.in_slices)

            # If length of the interval times is zero, turn off interval diagnostics
            if len(self.in_slices) == 0:
                for key in interval_dict:
                    interval_dict[key] = False

        # Validate tolerance to prevent overlapping intervals
        if len(self.in_slices) > 1:
            # Check regular gaps between adjacent points
            regular_gaps = np.diff(self.in_slices)

            # Check wrap-around gap (between last point and first point in next period)
            wrap_gap = (self.in_slices[0] + 1.0) - self.in_slices[-1]

            # Find the minimum gap considering both regular and wrap-around cases
            all_gaps = np.append(regular_gaps, wrap_gap)
            min_gap = np.min(all_gaps)

            # Convert to absolute time and calculate max allowed tolerance
            max_allowed_tolerance = min_gap

            if self.interval_tolerance >= max_allowed_tolerance:
                print(f"WARNING: interval_tolerance ({self.interval_tolerance:.4f}) is too large compared to the minimum gap between interval times.")
                print(f"Setting interval_tolerance to {max_allowed_tolerance * 0.9:.4f} to prevent overlapping intervals.")
                print(f"Minimum gap is {min_gap:.4f} fractions of a period.")
                self.interval_tolerance = max_allowed_tolerance * 0.9

        self.num_outputs = simulation_obj.num_diag_steps
        self.diag_folder = os.path.abspath(diag_outfolder)

        # Assemble master diagnostic dictionary
        self.master_diagnostic_dict = {
            'ieadfs': ieadfs,
            'eeadfs': eeadfs,
            'time_averaged': time_averaged_dict,
            'time_resolved': time_resolved_dict,
            'interval': interval_dict,
            'collisional': collisional_dict
        }

        # Import boundaries for edfs
        self.edf_bounds = np.array([])
        if hasattr(simulation_obj, 'edf_boundaries'):
            self.edf_bounds = np.array(simulation_obj.edf_boundaries)
            if any(self.edf_bounds < simulation_obj.zmin) or any(self.edf_bounds > simulation_obj.zmax):
                raise ValueError('simulation_obj.edf_boundaries ERROR: EDF boundaries must be within the range [zmin, zmax].')
            if not all(self.edf_bounds[i] < self.edf_bounds[i + 1] for i in range(len(self.edf_bounds) - 1)):
                raise ValueError('simulation_obj.edf_boundaries ERROR: EDF boundaries must be in ascending order.')

        # Set dictionaries of charge, mass, and collection array indices for each species
        self._make_particle_dictionaries()

        # Set up diagnostics
        self._import_general_timing_info(simulation_obj)
        self._get_time_resolved_steps(simulation_obj)
        self._setup_time_averaged_steps(simulation_obj)
        if any(interval_dict.values()):
            self._get_interval_collection_steps()
        else:
            self.in_coll_steps = [[] for _ in range(self.num_outputs)]
            self.step_to_interval_map = [{} for _ in range(self.num_outputs)]
            self.in_coll_counts = [{(0, i): 0 for i in range(len(self.in_slices))} for _ in range(self.num_outputs)]
        self._calculate_N_collections()
        self._setup_diagnostic_arrays(simulation_obj)

        # Save settings to file
        self._save_diagnostic_inputs()
        self._save_edf_settings()
        self._save_cells_and_nodes(simulation_obj)

        # Set diagnostic output indices
        self.curr_diag_output = 0
        self.curr_tr = 0
        self.curr_interval = 0
        self.curr_slice = 0

        # Save shared diagnostic variables for loop speedup
        self._save_shared_variables()

    ###########################################################################
    # Initialization Functions                                                #
    ###########################################################################
    def _setup_diagnostic_arrays(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Initialize diagnostic arrays

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        # Create ieadf bins
        self.ieadf_bin_edges = np.linspace(0, simulation_obj.ieadf_max_eV, simulation_obj.num_bins_ieadf + 1)
        self.ieadf_bin_centers = np.multiply(self.ieadf_bin_edges[:-1] + self.ieadf_bin_edges[1:], 0.5)
        self.iadf_bin_edges = np.linspace(-90, 90, 720 + 1)
        self.iadf_bin_centers = np.multiply(self.iadf_bin_edges[:-1] + self.iadf_bin_edges[1:], 0.5)

        # Ieadf arrays
        self.wall_eadf_by_species = {}
        for species in self.species_names[1:]:
            self.wall_eadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['ieadfs'].items():
                if value:
                    self.wall_eadf_by_species[species][key] = np.zeros((len(self.ieadf_bin_centers), len(self.iadf_bin_centers)))

        # Create eeadf bins
        self.eeadf_bin_edges = np.linspace(0, simulation_obj.eeadf_max_eV, simulation_obj.num_bins_eeadf + 1)
        self.eeadf_bin_centers = np.multiply(self.eeadf_bin_edges[:-1] + self.eeadf_bin_edges[1:], 0.5)
        self.eadf_bin_edges = np.linspace(-90, 90, 720 + 1)
        self.eadf_bin_centers = np.multiply(self.eadf_bin_edges[:-1] + self.eadf_bin_edges[1:], 0.5)

        # Eeadf arrays
        for species in [self.electron_name]:
            self.wall_eadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['eeadfs'].items():
                if value:
                    self.wall_eadf_by_species[species][key] = np.zeros((len(self.eeadf_bin_centers), len(self.eadf_bin_centers)))

        # Time resolved arrays - dictionary-based storage by species name
        self.tr_N = {key.replace('N_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                     for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('N_')}
        self.tr_W = {key.replace('W_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                     for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('W_')}
        self.tr_Wx = {key.replace('Wx_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Wx_')}
        self.tr_Wy = {key.replace('Wy_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Wy_')}
        self.tr_Wz = {key.replace('Wz_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Wz_')}
        self.tr_Jz = {key.replace('Jz_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Jz_')}
        self.tr_Jy = {key.replace('Jy_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Jy_')}
        self.tr_Jx = {key.replace('Jx_', ''): np.zeros((self.tr_coll[0], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Jx_')}
        self.tr_P_C = {key.replace('P_C_', ''): np.zeros((self.tr_coll[0], self.nz))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('P_C_')}
        self.tr_P_I = {key.replace('P_I_', ''): np.zeros((self.tr_coll[0], self.nz))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('P_I_')}
        self.tr_Pw = {key.replace('Pw_', ''): np.zeros((self.tr_coll[0], 2))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Pw_')}

        # Field diagnostics
        self.tr_E = {
            'z': np.zeros((self.tr_coll[0], self.nz)),
            'y': np.zeros((self.tr_coll[0], self.nz + 1)),
            'x': np.zeros((self.tr_coll[0], self.nz + 1))
        }
        self.tr_phi = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_J_d = np.zeros((self.tr_coll[0], self.nz))
        self.tr_J_w = np.zeros((self.tr_coll[0], 2))

        # Distribution functions by species
        self.tr_EDF = {key.replace('EDF_', ''): np.zeros((self.tr_coll[0], len(self.edf_bounds) + 1, len(self.edf_centers_by_species[key.replace('EDF_', '')])))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('EDF_')}
        self.tr_EVDF = {key: np.zeros((self.tr_coll[0], len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('ExDF_')}
        self.tr_EVDF.update({key: np.zeros((self.tr_coll[0], len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('EyDF_')})
        self.tr_EVDF.update({key: np.zeros((self.tr_coll[0], len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('EzDF_')})

        self.tr_times = np.zeros((self.tr_coll[0]))

        # Power arrays
        self.tr_Pin_vst = None

        # Time averaged arrays - dictionary-based storage by species name
        self.ta_N = {key.replace('N_', ''): np.zeros(self.nz + 1)
                     for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('N_')}
        self.ta_W = {key.replace('W_', ''): np.zeros(self.nz + 1)
                     for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('W_')}
        self.ta_W_collection_mask = {key.replace('W_', ''): np.zeros(self.nz + 1)
                                     for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('W_')}
        self.ta_Wx = {key.replace('Wx_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wx_')}
        self.ta_Wx_collection_mask = {key.replace('Wx_', ''): np.zeros(self.nz + 1)
                                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wx_')}
        self.ta_Wy = {key.replace('Wy_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wy_')}
        self.ta_Wy_collection_mask = {key.replace('Wy_', ''): np.zeros(self.nz + 1)
                                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wy_')}
        self.ta_Wz = {key.replace('Wz_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wz_')}
        self.ta_Wz_collection_mask = {key.replace('Wz_', ''): np.zeros(self.nz + 1)
                                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wz_')}
        self.ta_Jz = {key.replace('Jz_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Jz_')}
        self.ta_Jy = {key.replace('Jy_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Jy_')}
        self.ta_Jx = {key.replace('Jx_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Jx_')}
        self.ta_P_C = {key.replace('P_C_', ''): np.zeros(self.nz)
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('P_C_')}
        self.ta_P_I = {key.replace('P_I_', ''): np.zeros(self.nz)
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('P_I_')}
        self.ta_Pw = {key.replace('Pw_', ''): np.zeros(2)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Pw_')}

        # Field diagnostics
        self.ta_E = {
            'z': np.zeros(self.nz),
            'y': np.zeros(self.nz + 1),
            'x': np.zeros(self.nz + 1)
        }
        self.ta_phi = np.zeros(self.nz + 1)
        self.ta_J_d = np.zeros(self.nz)
        self.ta_J_w = np.zeros(2)

        # Distribution functions
        self.ta_EDF = {key.replace('EDF_', ''): np.zeros((len(self.edf_bounds) + 1, len(self.edf_centers_by_species[key.replace('EDF_', '')])))
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('EDF_')}
        self.ta_EVDF = {key: np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('ExDF_')}
        self.ta_EVDF.update({key: np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('EyDF_')})
        self.ta_EVDF.update({key: np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('EzDF_')})

        # Interval arrays - dictionary-based storage by species name
        self.in_N = {key.replace('N_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                     for key in self.master_diagnostic_dict['interval'] if key.startswith('N_')}
        self.in_W = {key.replace('W_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                     for key in self.master_diagnostic_dict['interval'] if key.startswith('W_')}
        self.in_W_collection_mask = {key.replace('W_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                     for key in self.master_diagnostic_dict['interval'] if key.startswith('W_')}
        self.in_Wx = {key.replace('Wx_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wx_')}
        self.in_Wx_collection_mask = {key.replace('Wx_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wx_')}
        self.in_Wy = {key.replace('Wy_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wy_')}
        self.in_Wy_collection_mask = {key.replace('Wy_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wy_')}
        self.in_Wz = {key.replace('Wz_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wz_')}
        self.in_Wz_collection_mask = {key.replace('Wz_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wz_')}
        self.in_Jz = {key.replace('Jz_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Jz_')}
        self.in_Jy = {key.replace('Jy_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Jy_')}
        self.in_Jx = {key.replace('Jx_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Jx_')}
        self.in_P_C = {key.replace('P_C_', ''): np.zeros((len(self.in_slices), self.nz))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('P_C_')}
        self.in_P_I = {key.replace('P_I_', ''): np.zeros((len(self.in_slices), self.nz))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('P_I_')}
        self.in_Pw = {key.replace('Pw_', ''): np.zeros((len(self.in_slices), 2))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Pw_')}

        # Field diagnostics
        self.in_E = {
            'z': np.zeros((len(self.in_slices), self.nz)),
            'y': np.zeros((len(self.in_slices), self.nz + 1)),
            'x': np.zeros((len(self.in_slices), self.nz + 1))
        }
        self.in_phi = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_J_d = np.zeros((len(self.in_slices), self.nz))
        self.in_J_w = np.zeros((len(self.in_slices), 2))

        # Distribution functions
        self.in_EDF = {key.replace('EDF_', ''): np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.edf_centers_by_species[key.replace('EDF_', '')])))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('EDF_')}
        self.in_EVDF = {key: np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                        for key in self.master_diagnostic_dict['interval'] if key.startswith('ExDF_')}
        self.in_EVDF.update({key: np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['interval'] if key.startswith('EyDF_')})
        self.in_EVDF.update({key: np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['interval'] if key.startswith('EzDF_')})

        # Dictionaries of single step collection arrays
        self.N = {}
        self.W = {}
        self.W_collection_mask = {}
        self.Wx = {}
        self.Wx_collection_mask = {}
        self.Wy = {}
        self.Wy_collection_mask = {}
        self.Wz = {}
        self.Wz_collection_mask = {}
        self.Jz = {}
        self.Jy = {}
        self.Jx = {}
        self.J_d = {}
        self.P_C = {}
        self.P_I = {}
        self.Pw = {}
        self.Edf = {}
        self.Exdf = {}
        self.Eydf = {}
        self.Ezdf = {}
        for species in self.species_names:
            for diag in self.PARTICLE_DIAGNOSTIC_PREFIXES:
                if any(dict.get(f'{diag}_{species}', False) for dict in [self.master_diagnostic_dict['time_averaged'], self.master_diagnostic_dict['time_resolved'], self.master_diagnostic_dict['interval']]):
                    if diag == 'N':
                        self.N[species] = np.zeros(self.nz + 1)
                    elif diag == 'W':
                        self.W[species] = np.zeros(self.nz + 1)
                    elif diag == 'Wx':
                        self.Wx[species] = np.zeros(self.nz + 1)
                    elif diag == 'Wy':
                        self.Wy[species] = np.zeros(self.nz + 1)
                    elif diag == 'Wz':
                        self.Wz[species] = np.zeros(self.nz + 1)
                    elif diag == 'Jz':
                        self.Jz[species] = np.zeros(self.nz + 1)
                    elif diag == 'Jy':
                        self.Jy[species] = np.zeros(self.nz + 1)
                    elif diag == 'Jx':
                        self.Jx[species] = np.zeros(self.nz + 1)
                    elif diag == 'P_C':
                        self.P_C[species] = np.zeros(self.nz)
                    elif diag == 'P_I':
                        self.P_I[species] = np.zeros(self.nz)
                    elif diag == 'Pw':
                        self.Pw[species] = np.zeros(2)
                    elif diag == 'EDF':
                        self.Edf[species] = np.zeros((len(self.edf_bounds) + 1, len(self.edf_centers_by_species[species])))
                    elif diag == 'ExDF':
                        self.Exdf[species] = np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[f'ExDF_{species}'])))
                    elif diag == 'EyDF':
                        self.Eydf[species] = np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[f'EyDF_{species}'])))
                    elif diag == 'EzDF':
                        self.Ezdf[species] = np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[f'EzDF_{species}'])))

        # Field diagnostics are always initialized since their number is small
        self.J_w = np.zeros(2)
        self.E = {
            'z': np.zeros(self.nz),
            'y': np.zeros(self.nz + 1),
            'x': np.zeros(self.nz + 1)
        }
        self.phi = np.zeros(self.nz + 1)
        self.E_z_last_step = np.zeros(self.nz)

    def _save_shared_variables(self):
        '''
        Save shared diagnostic variables to speed up do_diagnostics
        '''
        # Electric field
        self._Ez_wrapper = fields.EzFPWrapper()
        self._current_Ez_data = np.zeros(self.nz)

        self._Ey_wrapper = fields.EyFPWrapper()
        self._current_Ey_data = np.zeros(self.nz + 1)

        self.VELOCITY_SYNC_PREFIXES = ('Jz_', 'Jy_', 'Jx_', 'W_', 'Wx_', 'Wy_', 'Wz_')
        # Power prefixes must be computed before velocity synchronization so that
        # v^{n+1/2} is paired with E^n (both available at beforeEsolve time).
        # Synchronizing first would push v by +dt/2 using E^n, adding a spurious
        # O(dt) term to the J·E power estimate.
        # Note: this ordering requirement is only needed for update_P_C/update_P_I
        # (still used for time_resolved/interval P_C/P_I). The time_averaged P_C/P_I
        # diagnostics instead read WarpX's own per-push power deposition tracking
        # buffer (see _get_time_averaged_power_from_buffer), which captures the
        # exact velocity/field values used in the true numerical push and needs
        # no Python-side synchronization workaround.
        self.POWER_PREFIXES = ('P_C_', 'P_I_')

        # Diagnostic updates
        self.FIELD_DISPATCH = {
            'E_z': lambda: self.update_E('z'),
            'E_y': lambda: self.update_E('y'),
            'E_x': lambda: self.update_E('x'),
            'phi': self.update_phi,
            'J_d': self.update_J_d,
            'J_w': self.update_J_w,
        }
        self.SPECIES_DISPATCH = {
            'N_': self.update_N,
            'W_': self.update_W,
            'Wx_': lambda species, d='x': self.update_Wdir(species, d),
            'Wy_': lambda species, d='y': self.update_Wdir(species, d),
            'Wz_': lambda species, d='z': self.update_Wdir(species, d),
            'Jz_': lambda species, d='z': self.update_Jdir(species, d),
            'Jy_': lambda species, d='y': self.update_Jdir(species, d),
            'Jx_': lambda species, d='x': self.update_Jdir(species, d),
            'P_C_': self.update_P_C,
            'P_I_': self.update_P_I,
            'Pw_': self.update_Pw,
            'EDF_': self.calculate_edf,
            'ExDF_': lambda species: self.calculate_evdf(species, 'x'),
            'EyDF_': lambda species: self.calculate_evdf(species, 'y'),
            'EzDF_': lambda species: self.calculate_evdf(species, 'z'),
        }
        self.EVDF_PREFIXES = ('ExDF', 'EyDF', 'EzDF')

        self.collision_wrapper = collision_trackers.CollisionBufferWrapper()
        self.power_wrapper = power_deposition_trackers.PowerDepositionTrackerWrapper()

    def _calculate_N_collections(self):
        '''
        Calculate the number of collections for time averaged and resolved
        diagnostics at each diagnostic output.
        '''
        # Make arrays of length(num_outputs) for each diagnostic type
        self.tr_coll = np.zeros(self.num_outputs, dtype=int)
        self.ta_coll = np.zeros(self.num_outputs, dtype=int)

        # Calculate the number of collections for each diagnostic type
        # (for interval collections this is the number of collection
        #  intervals in each diagnostic output)
        for ii in range(self.num_outputs):
            total_steps = self.diag_stop[ii] - self.diag_start[ii]
            self.tr_coll[ii] = int((total_steps // self.diag_time_resolving_steps) + 1)
            self.ta_coll[ii] = int((total_steps // self.diag_time_averaging_steps) + 1)

        if comm.rank != 0:
            return

        # Check if the diagnostic folder exists
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)

        # Save the number of collections to file
        self._check_file(f'{self.diag_folder}/N_collections.dat')
        with open(f'{self.diag_folder}/N_collections.dat', 'w') as f:
            f.write('Number of Collections\n')
            f.write('---------------------\n')
            f.write('Diagnostic Output, Time Resolved, Time Averaged\n')
            for ii in range(self.num_outputs):
                f.write(f'{ii}, {self.tr_coll[ii]}, {self.ta_coll[ii]}\n')

    def _import_general_timing_info(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Import diagnostic steps for diagnostics.

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        # Import simulation parameters
        if self.restart_checkpoint:
            self.max_time = self.sim_ext.warpx.gett_new(lev=0) + simulation_obj.total_time
        else:
            self.max_time = simulation_obj.total_time

        self.diag_time         = simulation_obj.diag_time
        self.evolve_time       = simulation_obj.evolve_time

        self.diag_start        = simulation_obj.diag_start
        self.diag_stop         = simulation_obj.diag_stop

        self.diag_start_time   = self.diag_start[0] * self.dt

    def _get_interval_collection_steps(self):
        '''
        Set up arrays containing steps to calculate interval diagnostics.

        This method creates:
        1. self.in_coll_steps: A list of dictionaries where keys are (interval_idx, slice_idx)
           tuples and values are lists of time steps for each interval.
        2. self.in_coll_counts: A list of dictionaries tracking the number of time steps
           for each interval.
        3. self.step_to_interval_map: A list of dictionaries mapping time steps directly
           to their corresponding interval keys for fast lookup.

        Example
        -------
        Suppose we have 3 diagnostic outputs and can fit 4 intervals
        within each diagnostic window. Then the list will obey:

        ```
        len(self.in_coll_steps) = 3
        ```

        and for `ii` in `[0, 1, 2]`, `self.in_coll_steps[ii]` will be a dictionary
        mapping interval keys to lists of steps.

        For fast lookup, we also create `self.step_to_interval_map[ii]` which maps
        each step directly to its interval key.
        '''
        self.in_coll_steps = []
        self.in_coll_counts = []
        self.step_to_interval_map = []  # Map from step number to interval key

        for ii in range(self.num_outputs):
            # Start time of current diag output window
            output_start_t = self.diag_start[ii] * self.dt
            output_end_t = self.diag_stop[ii] * self.dt

            # Initialize collection times dictionary
            # Keys are interval indices, values are lists of collection times
            collection_times_dict = {}
            collection_counts_dict = {}
            step_mapping = {}  # Maps steps to interval keys

            # Count number of periods before the diagnostic output
            period_start_collection = int(output_start_t // self.in_period)

            # Get exact collection times
            # NOTE: We only collect for intervals that are fully within the diagnostic output
            exact_collec_times = (period_start_collection + self.in_slices) * self.in_period
            while any(exact_collec_times < output_start_t):
                exact_collec_times += self.in_period
                if any(exact_collec_times > output_end_t):
                    self.in_coll_steps.append({})
                    self.in_coll_counts.append({})
                    self.step_to_interval_map.append({})
                    break

            # Initialize the collection times dictionary
            interval_idx = 0

            # Calculate tolerance window half-width
            tolerance_half_width = self.in_period * self.interval_tolerance / 2.0

            while exact_collec_times[-1] <= output_end_t:
                # For each interval in this period
                for jj, exact_time in enumerate(exact_collec_times):
                    # Special case for zero tolerance - only take the single closest step
                    if self.interval_tolerance == 0.0:
                        # Find the closest step to exact_time
                        closest_step = int(round(exact_time / self.dt))
                        closest_step = max(self.diag_start[ii], min(self.diag_stop[ii], closest_step))
                        steps_in_window = [closest_step]
                    else:
                        # Calculate tolerance window
                        window_start = exact_time - tolerance_half_width
                        window_end = exact_time + tolerance_half_width

                        # Skip if window completely outside output window
                        if window_end < output_start_t or window_start > output_end_t:
                            continue

                        # Clamp window to output window
                        window_start = max(window_start, output_start_t)
                        window_end = min(window_end, output_end_t)

                        # Calculate all steps that fall within the window
                        start_step = max(self.diag_start[ii], int(np.floor(window_start / self.dt)))
                        end_step = min(self.diag_stop[ii], int(np.ceil(window_end / self.dt)))
                        steps_in_window = list(range(start_step, end_step + 1))

                    # Store steps in the dictionary with the interval index as key
                    interval_key = (interval_idx, jj)
                    collection_times_dict[interval_key] = steps_in_window
                    collection_counts_dict[interval_key] = len(steps_in_window)

                    # Create a mapping from step number to interval key for fast lookup
                    for step in steps_in_window:
                        # If a step is already mapped to an interval, choose the interval
                        # whose exact collection time is closest to the step time
                        if step in step_mapping:
                            # Calculate time difference for current interval
                            current_diff = abs(exact_time - step * self.dt)

                            # Calculate time difference for previously mapped interval
                            prev_interval_idx, prev_slice_idx = step_mapping[step]
                            prev_exact_time = (period_start_collection + prev_interval_idx) * self.in_period + self.in_slices[prev_slice_idx] * self.in_period
                            prev_diff = abs(prev_exact_time - step * self.dt)

                            # Keep mapping with smallest time difference
                            if current_diff < prev_diff:
                                step_mapping[step] = interval_key
                        else:
                            step_mapping[step] = interval_key

                # Move to next period
                exact_collec_times += self.in_period
                interval_idx += 1

            # Add the steps and counts to the collection arrays
            self.in_coll_steps.append(collection_times_dict)
            self.in_coll_counts.append(collection_counts_dict)
            self.step_to_interval_map.append(step_mapping)        # Save the interval collection steps to file
        if comm.rank != 0:
            return

        # Check if the folder exists
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)
        file = os.path.join(self.diag_folder, 'intrvl_collection_steps.dat')
        with open(file, 'w') as f:
            f.write('Interval Collection Steps\n')
            f.write(f'Interval Tolerance: {self.interval_tolerance}\n')
            for ii in range(self.num_outputs):
                f.write(f'\nDiagnostic Output #{ii+1}\n')
                f.write(f'---------------------\n')
                for (interval_idx, slice_idx), steps in self.in_coll_steps[ii].items():
                    f.write(f'Interval #{interval_idx+1}, Slice #{slice_idx+1}:\n')
                    f.write(f'    Steps: {steps}\n')
                    f.write(f'    Count: {self.in_coll_counts[ii][(interval_idx, slice_idx)]}\n')

            f.write('\nOptimized Step Mapping Info\n')
            f.write('-------------------------\n')
            total_steps = 0
            total_mapped_steps = 0

            # Track steps that were reassigned due to overlap
            overlap_count = 0

            for ii in range(self.num_outputs):
                if self.diag_stop[ii] > self.diag_start[ii]:
                    steps_in_output = self.diag_stop[ii] - self.diag_start[ii] + 1
                    mapped_steps = len(self.step_to_interval_map[ii])
                    total_steps += steps_in_output
                    total_mapped_steps += mapped_steps

                    # Check for overlapping intervals by counting steps in each interval
                    all_interval_steps = []
                    for steps_list in self.in_coll_steps[ii].values():
                        all_interval_steps.extend(steps_list)

                    # Count steps that appear multiple times
                    step_counts = {}
                    for step in all_interval_steps:
                        if step in step_counts:
                            step_counts[step] += 1
                        else:
                            step_counts[step] = 1

                    # Count overlapping steps
                    overlapping_steps = sum(1 for count in step_counts.values() if count > 1)
                    overlap_count += overlapping_steps

                    if steps_in_output > 0:
                        mapping_percent = (mapped_steps / steps_in_output) * 100
                    else:
                        mapping_percent = 0
                    f.write(f'Output #{ii+1}: {mapped_steps}/{steps_in_output} steps mapped ({mapping_percent:.2f}%)\n')
                    if overlapping_steps > 0:
                        f.write(f'  {overlapping_steps} steps were in multiple intervals and assigned to closest match\n')

            if total_steps > 0:
                total_percent = (total_mapped_steps / total_steps) * 100
            else:
                total_percent = 0
            f.write(f'Total: {total_mapped_steps}/{total_steps} steps mapped ({total_percent:.2f}%)\n')
            if overlap_count > 0:
                f.write(f'Total overlapping steps: {overlap_count} (assigned to closest interval)\n')

    def _get_time_resolved_steps(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Get step numbers to perform time resolved diagnostics. Computes:
        - self.num_in_tr: number of time resolved diagnostic collections per
          diagnostic output
        - self.tr_interval: time between time resolved diagnostic collections
        - self.diag_time_resolving_steps: number of steps between time resolved
          diagnostic collections

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation
        '''
        # Note: We calculate times in this function in seconds and then
        #       convert to time steps to get the most accurate step numbers

        # Import simulation parameters
        self.num_in_tr = simulation_obj.collections_per_diag_step
        if self.num_in_tr > int(self.diag_time / self.dt):
            self.num_in_tr = int(self.diag_time / self.dt)

        # Get time between time resolved diagnostic collections
        self.tr_interval = self.diag_time / self.num_in_tr

        # Convert times to steps
        self.diag_time_resolving_steps = int(self.tr_interval / self.dt)

    def _setup_time_averaged_steps(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Set up the time averaged diagnostic steps
        '''
        # Import time average collection parameters
        # If steps_bw_avg_collections is not included, default to 1
        if not hasattr(simulation_obj, 'steps_bw_avg_collections'):
            self.diag_time_averaging_steps = 1
        else:
            if simulation_obj.steps_bw_avg_collections <= 0 or not isinstance(simulation_obj.steps_bw_avg_collections, int):
                raise ValueError('steps_bw_avg_collections must be greater than zero.')
            self.diag_time_averaging_steps = simulation_obj.steps_bw_avg_collections

    def _save_diagnostic_inputs(self):
        '''
        Save diagnostic times and information to file
        '''
        if comm.rank != 0:
            return
        # Check if the folder exists
        if not os.path.exists(self.diag_folder) and comm.rank == 0:
            os.makedirs(self.diag_folder)
        file = os.path.join(self.diag_folder, 'diagnostic_times.dat')
        with open(file, 'w') as f:
            f.write('Simualtion Parameters\n')
            f.write('---------------------\n')
            f.write(f'Timestep [s]={self.dt}\n')
            f.write(f'Cell size [m]={self.dz}\n')
            f.write(f'Species: {", ".join(self.species_names)}\n\n')

            f.write('Diagnostic Parameters\n')
            f.write('---------------------\n')
            f.write(f'Diagnostics start time [s]={self.diag_start_time}\n')
            f.write(f'Diagnostic time [s]={self.diag_time}\n')
            f.write(f'Evolve time [s]={self.evolve_time}\n\n')

            f.write(f'Number of diagnostic outputs={self.num_outputs}\n\n')

            f.write(f'Time [s] between time resolved collections={self.tr_interval}\n')
            f.write(f'Time resolved collections per diagnostic={self.num_in_tr}\n\n')

            f.write(f'Number of steps between time average collections={self.diag_time_averaging_steps}\n\n')

            f.write(f'Interval period [s]={self.in_period}\n')
            f.write(f"Times in interval={', '.join(map(str,self.in_slices))}\n")
            f.write(f'Interval tolerance={self.interval_tolerance}\n\n')

            f.write(f'Output #   |   Start Step   |   Stop Step   |   Start Time   |   Stop Time\n')
            f.write(f'------------------------------------------------------------------------------\n')
            for ii in range(self.num_outputs):
                f.write(f'   {ii+1:5d}   | {self.diag_start[ii]:12d}   |{self.diag_stop[ii]:12d}   | {self.diag_start[ii]*self.dt:.8e} | {self.diag_stop[ii]*self.dt:.8e}\n')

            if self.edf_bounds is not None:
                f.write(f'\nEDF Boundaries [m]: {self.edf_bounds}\n')

    def _save_edf_settings(self):
        '''
        Save the settings for energy distribution function creation
        '''
        if comm.rank != 0:
            return

        # Make a diagnostics directory
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)

        # Save the wall EADF settings
        if any(self.master_diagnostic_dict['ieadfs'].values()):
            # Make an eadf directory for each ion species
            self.wall_eadf_dir_by_species = {}
            for species in self.species_names[1:]:
                self.wall_eadf_dir_by_species[species] = os.path.join(self.diag_folder, f'eadf_{species}')
                if not os.path.exists(self.wall_eadf_dir_by_species[species]):
                    os.makedirs(self.wall_eadf_dir_by_species[species])

            # Save the eadf energy bins
            for species in self.species_names[1:]:
                # Check if file exists
                self._check_file(f'{self.wall_eadf_dir_by_species[species]}/bins_eV.npy')
                self._check_file(f'{self.wall_eadf_dir_by_species[species]}/bins_deg.npy')
                np.save(f'{self.wall_eadf_dir_by_species[species]}/bins_eV.npy', self.ieadf_bin_centers)
                np.save(f'{self.wall_eadf_dir_by_species[species]}/bins_deg.npy', self.iadf_bin_centers)

        if any(self.master_diagnostic_dict['eeadfs'].values()):
            # Make an eeadf directory for electrons
            if not hasattr(self, 'wall_eadf_dir_by_species'):
                self.wall_eadf_dir_by_species = {}
            self.wall_eadf_dir_by_species[self.electron_name] = os.path.join(self.diag_folder, f'eadf_{self.electron_name}')
            if not os.path.exists(self.wall_eadf_dir_by_species[self.electron_name]):
                os.makedirs(self.wall_eadf_dir_by_species[self.electron_name])

            # Save the eeadf energy bins
            self._check_file(f"{self.wall_eadf_dir_by_species[self.electron_name]}/bins_eV.npy")
            self._check_file(f"{self.wall_eadf_dir_by_species[self.electron_name]}/bins_deg.npy")
            np.save(f"{self.wall_eadf_dir_by_species[self.electron_name]}/bins_eV.npy", self.eeadf_bin_centers)
            np.save(f"{self.wall_eadf_dir_by_species[self.electron_name]}/bins_deg.npy", self.eeadf_bin_centers)

        # Save the normal EDF settings
        for species in self.species_names:
            if any(dict.get(f'EDF_{species}', False) for dict in self.master_diagnostic_dict.values()):
                # Save the EDF energy bins
                self._check_file(f'{self.diag_folder}/edf_bins_eV_{species}.npy')
                np.save(f'{self.diag_folder}/edf_bins_eV_{species}.npy', self.edf_centers_by_species[species])
            if any(dict.get(f'ExDF_{species}', False) for dict in self.master_diagnostic_dict.values()):
                # Save the ExDF energy bins
                self._check_file(f'{self.diag_folder}/exdf_bins_eV_{species}.npy')
                np.save(f'{self.diag_folder}/exdf_bins_eV_{species}.npy', self.evdf_centers_by_diag_name[f'ExDF_{species}'])
            if any(dict.get(f'EyDF_{species}', False) for dict in self.master_diagnostic_dict.values()):
                # Save the EyDF energy bins
                self._check_file(f'{self.diag_folder}/eydf_bins_eV_{species}.npy')
                np.save(f'{self.diag_folder}/eydf_bins_eV_{species}.npy', self.evdf_centers_by_diag_name[f'EyDF_{species}'])
            if any(dict.get(f'EzDF_{species}', False) for dict in self.master_diagnostic_dict.values()):
                # Save the EzDF energy bins
                self._check_file(f'{self.diag_folder}/ezdf_bins_eV_{species}.npy')
                np.save(f'{self.diag_folder}/ezdf_bins_eV_{species}.npy', self.evdf_centers_by_diag_name[f'EzDF_{species}'])

    def _save_cells_and_nodes(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Save the cell boundaries and centers to file

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        if comm.rank != 0:
            return

        # Make a npy file of cell boundaries
        # Check if file exists
        self._check_file(f'{self.diag_folder}/nodes.npy')
        np.save(f'{self.diag_folder}/nodes.npy', self.nodes)

        # Make a npy file of cell centers
        z = np.linspace(self.dz / 2, simulation_obj.zmax - self.dz / 2, self.nz)
        # Check if file exists
        self._check_file(f'{self.diag_folder}/cells.npy')
        np.save(f'{self.diag_folder}/cells.npy', z)

    def _parse_species_controls_dict(self, switches: dict) -> tuple:
        '''
        Parse the new control dictionary format and extract species information.
        Build flat diagnostic dictionaries for internal use (backward compatibility).

        Parameters
        ----------
        switches: dict
            New format control dictionary with 'particle' and 'field' keys

        Returns
        -------
        tuple of (time_averaged_dict, time_resolved_dict, interval_dict, collisional_dict)
            Flat dictionaries matching the legacy format
        '''
        # Extract species from particles dict
        particles_dict = switches.get('particle', {})
        fields_dict = switches.get('field', {})

        # Build species_names and species_info
        self.species_names = []
        self.species_info = []

        for species_name, species_data in particles_dict.items():
            if '_' in species_name:
                error_msg = f"Species name '{species_name}' cannot contain underscores."
                raise ValueError(error_msg)
            self.species_names.append(species_name)

            # Extract properties for this species
            properties = species_data.get('properties', {})
            if 'm' not in properties and 'mass' not in properties:
                error_msg = f"Species '{species_name}' must have 'mass' ('m') key in 'properties' dict"
                raise ValueError(error_msg)

            if 'Z' not in properties and 'charge' not in properties:
                error_msg = f"Species '{species_name}' must have either 'Z' or 'charge' in 'properties' dict"
                raise ValueError(error_msg)

            if 'm' in properties and 'mass' not in properties:
                properties['mass'] = properties['m']
            if 'Z' not in properties and 'charge' in properties:
                charge_key = 'charge'
            else:
                charge_key = 'Z'

            self.species_info.append({
                'name': species_name,
                charge_key: properties[charge_key],
                'mass': properties['mass']
            })

        # Ensure electrons are first if present
        name_variants = ['electrons', 'e', 'e-']
        if any(x in self.species_names for x in name_variants):
            electron_idx = None
            for variant in name_variants:
                if variant in self.species_names:
                    electron_idx = self.species_names.index(variant)
                    break
            if electron_idx is not None and electron_idx != 0:
                # Move electrons to front
                self.species_names.insert(0, self.species_names.pop(electron_idx))
                self.species_info.insert(0, self.species_info.pop(electron_idx))

            # Save the electron name for frequent use
            self.electron_name = self.species_names[0]
        else:
            error_msg = "No electron species found (searching for one of 'electrons', 'e', 'e-').\n" \
                        "Without this, EDF diagnostics will not be correctly organized.\n" \
                        "Code would need to be rewritten to handle this case. Please ensure an\n" \
                        "electron species is included with one of the accepted names."
            raise ValueError(error_msg)

        # Build flat diagnostic dictionaries for each type
        time_averaged_dict = {}
        time_resolved_dict = {}
        interval_dict = {}

        # Process each species
        for species_name in self.species_names:
            species_data = particles_dict[species_name]

            # Append all species names to diagnostic
            suffix = f'_{species_name}'

            # Process time_averaged diagnostics
            ta_dict = species_data.get('time_averaged', {})
            for diag in self.PARTICLE_DIAGNOSTIC_PREFIXES:
                if ta_dict.get(diag, False):
                    time_averaged_dict[f'{diag}{suffix}'] = True

            # Process time_resolved diagnostics
            tr_dict = species_data.get('time_resolved', {})
            for diag in self.PARTICLE_DIAGNOSTIC_PREFIXES:
                if tr_dict.get(diag, False):
                    time_resolved_dict[f'{diag}{suffix}'] = True

            # Process interval diagnostics
            in_dict = species_data.get('interval', {})
            for diag in self.PARTICLE_DIAGNOSTIC_PREFIXES:
                if in_dict.get(diag, False):
                    interval_dict[f'{diag}{suffix}'] = True

        # Process collision diagnostics (top-level collision names, e.g. 'coll_elec', 'e_h3_recombination')
        collisional_dict = switches.get('collision', {})

        # Build EDF bins
        self.edf_edges_by_species = {}
        self.evdf_edges_by_diag_name = {}
        self.edf_centers_by_species = {}
        self.evdf_centers_by_diag_name = {}

        for species_name, species_data in particles_dict.items():
            properties = species_data.get('properties', {})

            # EDFs
            if any(dict.get(f'EDF_{species_name}', False) for dict in [time_averaged_dict, time_resolved_dict, interval_dict]):
                if 'num_bins_edf' in properties:
                    self.edf_edges_by_species[species_name] = np.linspace(0, properties['max_edf'], properties['num_bins_edf'] + 1)
                    self.edf_centers_by_species[species_name] = (self.edf_edges_by_species[species_name][:-1] + self.edf_edges_by_species[species_name][1:]) / 2

            # EVDFs
            for dir in ['x', 'y', 'z']:
                if any(dict.get(f'E{dir}DF_{species_name}', False) for dict in [time_averaged_dict, time_resolved_dict, interval_dict]):
                    if f'num_bins_e{dir}df' not in properties:
                        raise ValueError(f"Species '{species_name}' does not have 'num_bins_e{dir}df' key in 'properties'.")
                    if f'max_e{dir}df' not in properties:
                        raise ValueError(f"Species '{species_name}' does not have 'max_e{dir}df' key in 'properties'.")

                    diag_name = f'E{dir}DF_{species_name}'
                    self.evdf_edges_by_diag_name[diag_name] = np.linspace(-properties[f'max_e{dir}df'], properties[f'max_e{dir}df'], properties[f'num_bins_e{dir}df'] + 1)
                    self.evdf_centers_by_diag_name[diag_name] = (self.evdf_edges_by_diag_name[diag_name][:-1] + self.evdf_edges_by_diag_name[diag_name][1:]) / 2

        # Process field diagnostics
        for diag_type, target_dict in [('time_averaged', time_averaged_dict),
                                        ('time_resolved', time_resolved_dict),
                                        ('interval', interval_dict)]:
            field_dict = fields_dict.get(diag_type, {})
            for field_name in self.FIELD_DIAGNOSTICS:
                if field_dict.get(field_name, False):
                    target_dict[field_name] = True

        return time_averaged_dict, time_resolved_dict, interval_dict, collisional_dict

    def _make_particle_dictionaries(self):
        '''
        Make dictionaries with keys self.species_names for diag indices,
        mass, and charge.
        '''
        self.mass_by_name = {}
        self.charge_by_name = {}
        for info in self.species_info:
            self.mass_by_name[info['name']] = info['mass']

            # Determine charge
            if 'Z' in info:
                self.charge_by_name[info['name']] = info['Z'] * constants.q_e
            elif 'charge' in info:
                self.charge_by_name[info['name']] = info['charge']
            else:
                raise ValueError(f"Species '{info['name']}' does not have 'Z' or 'charge' keys.")

    ###########################################################################
    # Diagnostic Functions                                                    #
    ###########################################################################
    # def update_N(self, species):
    #     '''
    #     Return density [m^-3] at node points for a species. Needs be multiplied
    #     by charge to get charge density.

    #     Parameters
    #     ----------
    #     species: str
    #         Name of species
    #     '''
    #     # Set up wrappers
    #     rho_wrapper = fields.RhoFPWrapper()
    #     species_wrapper = particle_containers.ParticleContainerWrapper(species)
    #     species_wrapper.deposit_charge_density(level=0, clear_rho=True)

    #     # Report the density
    #     rho_data = rho_wrapper[...]
    #     self.N[species] = rho_data

    def update_N(self, species):
        '''
        Return density [m^-3] at node points for a species. Needs be
        divided by cell size before being used.

        Parameters
        ----------
        species: str
            Name of species
        '''
        # Set up wrappers
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle quantities
        try:
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            w = np.array([])
            z = np.array([])

        # Get cell index of particle
        cell_idx = np.floor(z / self.dz).astype(int)

        # Calculate the fractional position within the cell
        frac_pos = (z / self.dz) - cell_idx

        # Calculate weights for interpolation
        frac_l = 1 - frac_pos
        frac_r = frac_pos

        # Sort by z and assign w to nodes
        temp_N = np.zeros(self.nz + 1)
        np.add.at(temp_N, cell_idx, w * frac_l)
        # Get a list of all particles which are not at the last node
        valid_idxs = cell_idx != self.nz
        np.add.at(temp_N, cell_idx[valid_idxs] + 1, w[valid_idxs] * frac_r[valid_idxs])

        # Multiply the first and last element by 2 to account for the half cell
        temp_N[0] *= 2
        temp_N[-1] *= 2

        # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_N to all processes
        N_data = np.zeros_like(temp_N)
        comm.Allreduce(temp_N, N_data, op=mpi.SUM)

        # Report the current
        self.N[species] = N_data

    def update_W(self, species):
        '''
        Return average energy [eV] at node points for a species. Needs be multiplied
        by v2_factor = mass / (2.0 * 1.6e-19) before being used.

        Parameters
        ----------
        species: str
            Name of species
        '''
        # Set up wrappers
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle velocities
        try:
            ux = np.concatenate(species_wrapper.get_particle_ux())
            uy = np.concatenate(species_wrapper.get_particle_uy())
            uz = np.concatenate(species_wrapper.get_particle_uz())
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            ux = np.array([])
            uy = np.array([])
            uz = np.array([])
            w = np.array([])
            z = np.array([])

        # Get temperature (E = 0.5mv^2 = 1.5T)
        v2 = ux**2 + uy**2 + uz**2

        # Get cell index of particle
        cell_idx = np.floor(z / self.dz).astype(int)

        # Calculate the fractional position within the cell
        frac_pos = (z / self.dz) - cell_idx

        # Calculate weights for interpolation
        frac_l = 1 - frac_pos
        frac_r = frac_pos

        # Sort by z and assign w and W to nodes
        temp_W = np.zeros(self.nz + 1)
        temp_w = np.zeros(self.nz + 1)
        np.add.at(temp_W, cell_idx, v2 * w * frac_l)
        np.add.at(temp_w, cell_idx, w * frac_l)
        # Get a list of all particles which are not at the last node
        valid_idxs = cell_idx != self.nz
        np.add.at(temp_W, cell_idx[valid_idxs] + 1, v2[valid_idxs] * w[valid_idxs] * frac_r[valid_idxs])
        np.add.at(temp_w, cell_idx[valid_idxs] + 1, w[valid_idxs] * frac_r[valid_idxs])

       # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_W to all processes
        W_data = np.zeros_like(temp_W)
        w_data = np.zeros_like(temp_w)
        comm.Allreduce(temp_W, W_data, op=mpi.SUM)
        comm.Allreduce(temp_w, w_data, op=mpi.SUM)

        # Divide by weight to get average
        W_data = np.divide(W_data, w_data, out=np.zeros_like(W_data, dtype=float), where=w_data!=0)

        # Report the temperature
        self.W[species] = W_data

        # Get a truth value for whether the species is in a particular cell
        self.W_collection_mask[species] = (w_data != 0).astype(float)

    def update_Wdir(self, species, direction):
        '''
        Return average directional energy [eV] at node points for a species.
        v2 = u_direction**2, e.g. ux**2 for direction='x'.
        Needs to be multiplied by mass / (2 * q_e) before being used.

        Parameters
        ----------
        species: str
            Name of species
        direction: str
            Velocity component to use: 'x', 'y', or 'z'
        '''
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        try:
            u = np.concatenate(getattr(species_wrapper, f'get_particle_u{direction}')())
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            u = np.array([])
            w = np.array([])
            z = np.array([])

        v2 = u**2

        cell_idx = np.floor(z / self.dz).astype(int)
        frac_pos = (z / self.dz) - cell_idx
        frac_l = 1 - frac_pos
        frac_r = frac_pos

        temp_W = np.zeros(self.nz + 1)
        temp_w = np.zeros(self.nz + 1)
        np.add.at(temp_W, cell_idx, v2 * w * frac_l)
        np.add.at(temp_w, cell_idx, w * frac_l)
        valid_idxs = cell_idx != self.nz
        np.add.at(temp_W, cell_idx[valid_idxs] + 1, v2[valid_idxs] * w[valid_idxs] * frac_r[valid_idxs])
        np.add.at(temp_w, cell_idx[valid_idxs] + 1, w[valid_idxs] * frac_r[valid_idxs])

        W_data = np.zeros_like(temp_W)
        w_data = np.zeros_like(temp_w)
        comm.Allreduce(temp_W, W_data, op=mpi.SUM)
        comm.Allreduce(temp_w, w_data, op=mpi.SUM)

        W_data = np.divide(W_data, w_data, out=np.zeros_like(W_data, dtype=float), where=w_data!=0)

        getattr(self, f'W{direction}')[species] = W_data
        getattr(self, f'W{direction}_collection_mask')[species] = (w_data != 0).astype(float)

    def update_Jdir(self, species, direction):
        '''
        Return current density [A/m^2] at node points for a species along
        the specified direction. Needs to be multiplied by charge and divided
        by cell size before being used.

        Parameters
        ----------
        species: str
            Name of species
        direction: str
            Velocity component to use: 'x', 'y', or 'z'
        '''
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        try:
            u = np.concatenate(getattr(species_wrapper, f'get_particle_u{direction}')())
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            u = np.array([])
            w = np.array([])
            z = np.array([])

        cell_idx = np.floor(z / self.dz).astype(int)
        frac_pos = (z / self.dz) - cell_idx
        frac_l = 1 - frac_pos
        frac_r = frac_pos

        temp_J = np.zeros(self.nz + 1)
        np.add.at(temp_J, cell_idx, u * w * frac_l)
        valid_idxs = cell_idx != self.nz
        np.add.at(temp_J, cell_idx[valid_idxs] + 1, u[valid_idxs] * w[valid_idxs] * frac_r[valid_idxs])

        J_data = np.zeros_like(temp_J)
        comm.Allreduce(temp_J, J_data, op=mpi.SUM)

        getattr(self, f'J{direction}')[species] = J_data

    def update_J_w(self):
        '''
        Return current density [A/m^2] at the left and right boundaries
        for all species. Needs to be multiplied by charge and divided by
        dt before being used.
        '''
        buffer = particle_containers.ParticleBoundaryBufferWrapper()
        lev = 0

        # Initialize arrays for each boundary (float, since particle weights
        # are not generally integers)
        J_w_lo = np.zeros(1)
        J_w_hi = np.zeros(1)

        # Process z_lo boundary (only on rank 0)
        if comm.rank == 0:
            for species in self.species_names:
                try:
                    w = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_lo', "w", lev))
                    count = np.sum(w)
                except ValueError:
                    count = 0

                # Charge number of the species (e.g. -1 for electrons, +1 for singly charged ions)
                charge_number = self.charge_by_name[species] / constants.q_e

                # For z_lo, boundary_factor = -1
                J_w_lo[0] -= charge_number * count

            # Add SEE contribution
            if self.SEE_obj is not None:
                J_w_lo[0] -= self.SEE_obj.SEE_current_this_step['z_lo']

        # Process z_hi boundary (only on rank num_proc - 1)
        if comm.rank == num_proc - 1:
            for species in self.species_names:
                try:
                    w = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_hi', "w", lev))
                    count = np.sum(w)
                except ValueError:
                    count = 0

                # Charge number of the species (e.g. -1 for electrons, +1 for singly charged ions)
                charge_number = self.charge_by_name[species] / constants.q_e

                # For z_hi, boundary_factor = 1
                J_w_hi[0] += charge_number * count

            # Add SEE contribution
            if self.SEE_obj is not None:
                J_w_hi[0] += self.SEE_obj.SEE_current_this_step['z_hi']

        # Broadcast results to all processes
        comm.Bcast(J_w_lo, root=0)
        comm.Bcast(J_w_hi, root=num_proc - 1)

        # Save the results
        self.J_w[0] = J_w_lo[0]
        self.J_w[1] = J_w_hi[0]

    def update_Pw(self, species):
        '''
        Return kinetic energy [J] deposited at the left and right boundaries
        by particles of this species scraped this step. Needs to be divided
        by dt before being used, to get power [W].

        If an SEE object is provided and this is the electron species, the
        kinetic energy carried away by secondary electrons emitted from each
        boundary this step is subtracted, since that energy is drawn from the
        wall rather than deposited onto it.

        Parameters
        ----------
        species: str
            Name of species
        '''
        buffer = particle_containers.ParticleBoundaryBufferWrapper()
        lev = 0
        mass = self.mass_by_name[species]

        Pw_lo = np.zeros(1)
        Pw_hi = np.zeros(1)

        # Process z_lo boundary (only on rank 0)
        if comm.rank == 0:
            try:
                ux = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_lo', "ux", lev))
                uy = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_lo', "uy", lev))
                uz = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_lo', "uz", lev))
                w  = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_lo', "w", lev))
                Pw_lo[0] = 0.5 * mass * np.sum((ux**2 + uy**2 + uz**2) * w)
            except ValueError:
                pass

            # Subtract energy carried away by secondary electrons emitted from this boundary
            if self.SEE_obj is not None and species == self.electron_name:
                Pw_lo[0] -= self.SEE_obj.SEE_current_this_step['z_lo'] * self.SEE_obj.SEE_energy_J

        # Process z_hi boundary (only on rank num_proc - 1)
        if comm.rank == num_proc - 1:
            try:
                ux = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_hi', "ux", lev))
                uy = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_hi', "uy", lev))
                uz = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_hi', "uz", lev))
                w  = np.concatenate(buffer.get_particle_scraped_this_step(species, 'z_hi', "w", lev))
                Pw_hi[0] = 0.5 * mass * np.sum((ux**2 + uy**2 + uz**2) * w)
            except ValueError:
                pass

            # Subtract energy carried away by secondary electrons emitted from this boundary
            if self.SEE_obj is not None and species == self.electron_name:
                Pw_hi[0] -= self.SEE_obj.SEE_current_this_step['z_hi'] * self.SEE_obj.SEE_energy_J

        # Broadcast results to all processes
        comm.Bcast(Pw_lo, root=0)
        comm.Bcast(Pw_hi, root=num_proc - 1)

        # Save the results
        self.Pw[species][0] = Pw_lo[0]
        self.Pw[species][1] = Pw_hi[0]

    def update_E(self, direction: str):
        '''
        Return electric field at node points

        Parameters
        ----------
        direction: str
            Direction of electric field to update ('x', 'y', or 'z')
        '''
        match direction:
            case 'x':
                self.E['x'] = fields.ExFPWrapper()[...]
            case 'y':
                self.E['y'] = self._Ey_wrapper[...]
            case 'z':
                self.E['z'] = self._current_Ez_data

    def update_phi(self):
        '''
        Return potential at node points
        '''
        phi_wrapper = fields.PhiFPWrapper()
        self.phi = phi_wrapper[...]

    def update_P_I(self, species):
        '''
        Calculate power into plasma via a self-consistent ICP Field.
        Needs to be multiplied by charge and divided by cell size before
        being used.

        This interpolates the field to the particle positions using a linear
        shape, similar to what WarpX does. Minor differences in the two methods
        (e.g. I don't know exactly how WarpX interpolates for particles at the
        boundary) may lead to slight differences from the actual power.

        Note: only used for time_resolved/interval P_I diagnostics. Time-averaged
        P_I instead reads WarpX's own power deposition tracking buffer directly
        (see _get_time_averaged_power_from_buffer), which is exact rather than
        an approximate re-interpolation, and captures every push step instead
        of only the sampled diagnostic steps.
        '''
        # Set up wrappers
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle velocities
        try:
            uy = np.concatenate(species_wrapper.get_particle_uy())
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            uy = np.array([])
            w = np.array([])
            z = np.array([])

        # Field is on the nodes, so average it out to the cell centers
        Ey_centers = (self._current_Ey_data[:-1] + self._current_Ey_data[1:]) / 2

        # Get cell index of particles
        cell_idx = np.floor(z / self.dz).astype(int)

        # Calculate the fractional position within the cell
        frac_pos = (z / self.dz) - cell_idx

        # Initialize the array of the field at each particle position
        Ey_at_particle = np.zeros(len(z))

        # Create masks to classify the particles
        mask_low_edge = (cell_idx == 0) & (frac_pos <= 0.5)
        mask_high_edge = (cell_idx == self.nz - 1) & (frac_pos >= 0.5) | (cell_idx == self.nz)
        # Ensure that the edge cases are not picked up by the other masks
        mask_before_center = (frac_pos < 0.5) & ~(mask_low_edge | mask_high_edge)
        mask_after_center = (frac_pos >= 0.5) & ~(mask_low_edge | mask_high_edge)

        # Handle particles near the low edge (index 0)
        Ey_at_particle[mask_low_edge] = Ey_centers[0]

        # Handle particles near the high edge (index nz - 1) and
        # revert all cell indices at the high edge to the last cell (this
        # prevents out of bounds errors for particles exactly at the boundary)
        cell_idx[mask_high_edge] = self.nz - 1
        Ey_at_particle[mask_high_edge] = Ey_centers[self.nz - 1]

        # Handle particles before the center of the cell
        rel_position_before = frac_pos[mask_before_center] + 0.5
        Ey_at_particle[mask_before_center] = (
            Ey_centers[cell_idx[mask_before_center] - 1] +
            (Ey_centers[cell_idx[mask_before_center]] - Ey_centers[cell_idx[mask_before_center] - 1]) * rel_position_before
        )

        # Handle particles after the center of the cell
        rel_position_after = frac_pos[mask_after_center] - 0.5
        Ey_at_particle[mask_after_center] = (
            Ey_centers[cell_idx[mask_after_center]] +
            (Ey_centers[cell_idx[mask_after_center] + 1] - Ey_centers[cell_idx[mask_after_center]]) * rel_position_after
        )

        # # Commenting this out, but writing out how to do a linear interpolation
        # # for the external particles, incase I find out this is what WarpX does
        # first_position = 1.5 - frac_pos
        # Ey_at_particle[first_parts] = Ey_centers[1] - (Ey_centers[1] - Ey_centers[0]) * first_position
        # end_position = 0.5 + frac_pos
        # Ey_at_particle[end_parts] = Ey_centers[self.nz - 2] + (Ey_centers[self.nz - 1] - Ey_centers[self.nz - 2]) * end_position

        # Sort by z and assign power input to cells
        temp_P = np.zeros(self.nz)
        np.add.at(temp_P, cell_idx, uy * Ey_at_particle * w)

        # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_P to all processes
        P_data = np.zeros_like(temp_P)
        comm.Allreduce(temp_P, P_data, op=mpi.SUM)

        # Report the temperature
        self.P_I[species] = P_data

    def update_P_C(self, species):
        '''
        Calculate power into plasma via capacitive heating. Needs to be
        multiplied by charge and divided by cell size before being used.

        This interpolates the field to the particle positions using a linear
        shape, similar to what WarpX does. Minor differences in the two methods
        (e.g. I don't know exactly how WarpX interpolates for particles at the
        boundary) may lead to slight differences from the actual power.

        Note: only used for time_resolved/interval P_C diagnostics. Time-averaged
        P_C instead reads WarpX's own power deposition tracking buffer directly
        (see _get_time_averaged_power_from_buffer), which is exact rather than
        an approximate re-interpolation, and captures every push step instead
        of only the sampled diagnostic steps.
        '''
        # Set up wrappers
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle velocities
        try:
            uz = np.concatenate(species_wrapper.get_particle_uz())
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            uz = np.array([])
            w = np.array([])
            z = np.array([])

        # Get the perpendicular field (on the cell centers)
        Ez_centers = self._current_Ez_data

        # Get cell index of particles
        cell_idx = np.floor(z / self.dz).astype(int)

        # Calculate the fractional position within the cell
        frac_pos = (z / self.dz) - cell_idx

        # Initialize the array of the field at each particle position
        Ez_at_particle = np.zeros(len(z))

        # Create masks to classify the particles
        mask_low_edge = (cell_idx == 0) & (frac_pos <= 0.5)
        mask_high_edge = (cell_idx == self.nz - 1) & (frac_pos >= 0.5) | (cell_idx == self.nz)
        # Ensure that the edge cases are not picked up by the other masks
        mask_before_center = (frac_pos < 0.5) & ~(mask_low_edge | mask_high_edge)
        mask_after_center = (frac_pos >= 0.5) & ~(mask_low_edge | mask_high_edge)

        # Handle particles near the low edge (index 0)
        Ez_at_particle[mask_low_edge] = Ez_centers[0]

        # Handle particles near the high edge (index nz - 1) and
        # revert all cell indices at the high edge to the last cell (this
        # prevents out of bounds errors for particles exactly at the boundary)
        cell_idx[mask_high_edge] = self.nz - 1
        Ez_at_particle[mask_high_edge] = Ez_centers[self.nz - 1]

        # Handle particles before the center of the cell
        rel_position_before = frac_pos[mask_before_center] + 0.5
        Ez_at_particle[mask_before_center] = (
            Ez_centers[cell_idx[mask_before_center] - 1] +
            (Ez_centers[cell_idx[mask_before_center]] - Ez_centers[cell_idx[mask_before_center] - 1]) * rel_position_before
        )

        # Handle particles after the center of the cell
        rel_position_after = frac_pos[mask_after_center] - 0.5
        Ez_at_particle[mask_after_center] = (
            Ez_centers[cell_idx[mask_after_center]] +
            (Ez_centers[cell_idx[mask_after_center] + 1] - Ez_centers[cell_idx[mask_after_center]]) * rel_position_after
        )

        # # Commenting this out, but writing out how to do a linear interpolation
        # # for the external particles, incase I find out this is what WarpX does
        # first_position = 1.5 - frac_pos
        # Ex_at_particle[first_parts] = Ex_centers[1] - (Ex_centers[1] - Ex_centers[0]) * first_position
        # end_position = 0.5 + frac_pos
        # Ex_at_particle[end_parts] = Ex_centers[self.nz - 2] + (Ex_centers[self.nz - 1] - Ex_centers[self.nz - 2]) * end_position

        # Sort by z and assign power input to cells
        temp_P = np.zeros(self.nz)
        np.add.at(temp_P, cell_idx, uz * Ez_at_particle * w)

        # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_P to all processes
        P_data = np.zeros_like(temp_P)
        comm.Allreduce(temp_P, P_data, op=mpi.SUM)

        # Report the temperature
        self.P_C[species] = P_data

    def _get_time_averaged_power_from_buffer(self, species, direction):
        '''
        Read and clear WarpX's power deposition tracking buffer for a species,
        converting the raw accumulated buffer into a time-averaged power per
        unit length [W/m] suitable for accumulating into ta_P_C/ta_P_I.

        The buffer sums one w*q*v*E power sample per elapsed push step since
        it was last cleared (charge is already included by WarpX). Converting
        to a time-averaged power means: divide by cell size, then divide by
        the number of elapsed steps since the most recent diagnostic
        collection -- equivalently, multiply by the timestep and divide by
        the elapsed time since the most recent diagnostic collection, since
        that elapsed time is exactly diag_time_averaging_steps * dt.

        Requires the species to have enable_power_deposition_tracking=True.

        Parameters
        ----------
        species: str
            Species name
        direction: str
            'x', 'y', or 'z' -- which power component to return ('z' for
            capacitive/P_C, 'y' for inductive/P_I)

        Returns
        -------
        np.ndarray
            Time-averaged power per unit length for this collection window,
            shape (self.nz,). All zeros if the species has no particles.
        '''
        px, py, pz = self.power_wrapper.get(species, level=0, gather=True, normalize=False)
        raw = {'x': px, 'y': py, 'z': pz}[direction]

        # Always clear so the next collection window starts fresh, regardless
        # of whether this window had any particles/data.
        self.power_wrapper.clear_buffers([species], level=0)

        if raw is None:
            return np.zeros(self.nz)

        elapsed_time = self.diag_time_averaging_steps * self.dt
        return (raw / self.dz) * self.dt / elapsed_time

    def update_J_d(self):
        '''
        Calculate the displacement current density. Needs be multiplied by
        constants.ep0 and divided by the time step before being used.

        We use a backward difference to calculate the displacement current
        and CANNOT use this implementation at the first time step.
        '''
        # Save the electric field from the current time step, if not already done
        if not any(dict.get('E_z') for dict in self.master_diagnostic_dict.values()):
              self.E['z'] = self._current_Ez_data

        # Calculate the displacement current density
        self.J_d = self.E['z'] - self.E_z_last_step

    def calculate_edf(self, species: str):
        '''
        Gets an energy distribution function for the requested species.

        Parameters
        ----------
        species: str
            The name of the species

        Returns
        -------
        hist: np.ndarray
            The histogram of the energy distribution function
        '''
        # Get the edf on the processor
        hist = self._get_edf(species)

        # Sum the edf histograms from all processors
        hist_all = np.zeros_like(hist)
        comm.Allreduce(hist, hist_all, op=mpi.SUM)

        self.Edf[species] = hist_all

    def _get_edf(self, species):
        '''
        Gets an energy distribution function.
        '''
        # Set up wrapper
        species_wrapper = particle_containers.ParticleContainerWrapper(species)
        try:
            z  = np.concatenate(species_wrapper.get_particle_z())
            ux = np.concatenate(species_wrapper.get_particle_ux())
            uy = np.concatenate(species_wrapper.get_particle_uy())
            uz = np.concatenate(species_wrapper.get_particle_uz())
            w  = np.concatenate(species_wrapper.get_particle_weight())
        except ValueError:
            z  = np.array([])
            ux = np.array([])
            uy = np.array([])
            uz = np.array([])
            w  = np.array([])

        # Calculate the energy
        v2 = (np.square(ux) + np.square(uy) + np.square(uz))
        E = np.multiply(v2, 0.5 * self.mass_by_name[species] / constants.q_e)

        mask = np.zeros((len(self.edf_bounds) + 1, len(z)), dtype=bool)
        if len(self.edf_bounds) > 0:
            mask[0] = z < self.edf_bounds[0]
            for ii in range(1, len(self.edf_bounds)):
                mask[ii] = (z >= self.edf_bounds[ii-1]) & (z < self.edf_bounds[ii])
            mask[-1] = z >= self.edf_bounds[-1]
        else:
            mask[0] = np.ones_like(z, dtype=bool)

        hist_by_mask = []

        for i in range(len(self.edf_bounds) + 1):
            # Get the histogram (unnormalized)
            hist, _ = np.histogram(E[mask[i]], bins=self.edf_edges_by_species[species], density=False, weights=w[mask[i]] / self.dz)

            hist = np.copy(hist, order='C')
            hist_by_mask.append(hist)

        hist_by_mask = np.stack(hist_by_mask)

        return hist_by_mask

    def calculate_evdf(self, species: str, direction: str):
        '''
        Gets an energy distribution function for the requested species in a specific direction.

        Parameters
        ----------
        species: str
            The name of the species
        direction: str
            The velocity component of the distribution function, one of 'x', 'y', 'z'

        Returns
        -------
        hist: np.ndarray
            The histogram of the distribution function
        '''
        # Get the vdf on the processor
        hist = self._get_evdf(species, direction)

        # Sum the vdf histograms from all processors
        hist_all = np.zeros_like(hist)
        comm.Allreduce(hist, hist_all, op=mpi.SUM)

        if direction == 'x':
            self.Exdf[species] = hist_all
        elif direction == 'y':
            self.Eydf[species] = hist_all
        elif direction == 'z':
            self.Ezdf[species] = hist_all

    def _get_evdf(self, species, direction):
        '''
        Gets an energy distribution function.
        '''
        # Set up wrapper
        species_wrapper = particle_containers.ParticleContainerWrapper(species)
        try:
            z = np.concatenate(species_wrapper.get_particle_z())
            u = np.concatenate(species_wrapper.__getattribute__(f'get_particle_u{direction}')())
            w = np.concatenate(species_wrapper.get_particle_weight())
        except ValueError:
            z  = np.array([])
            u = np.array([])
            w  = np.array([])

        # Calculate the energy
        u2_signed = np.multiply(np.square(u), np.sign(u))
        E = np.multiply(u2_signed, 0.5 * self.mass_by_name[species] / constants.q_e)

        # Sort particles based on z position
        mask = np.zeros((len(self.edf_bounds) + 1, len(z)), dtype=bool)
        if len(self.edf_bounds) > 0:
            mask[0] = z < self.edf_bounds[0]
            for ii in range(1, len(self.edf_bounds)):
                mask[ii] = (z >= self.edf_bounds[ii-1]) & (z < self.edf_bounds[ii])
            mask[-1] = z >= self.edf_bounds[-1]
        else:
            mask[0] = np.ones_like(z, dtype=bool)

        hist_by_mask = []

        for i in range(len(self.edf_bounds) + 1):
            # Get the histogram (unnormalized)
            hist, _ = np.histogram(E[mask[i]], bins=self.evdf_edges_by_diag_name[f'E{direction}DF_{species}'], density=False, weights=w[mask[i]] / self.dz)

            hist = np.copy(hist, order='C')
            hist_by_mask.append(hist)

        hist_by_mask = np.stack(hist_by_mask)

        return hist_by_mask

    def calculate_wall_eadf(self, species: str, boundary: str):
        '''
        Gets a histogram of the energy angular distribution function at
        the specified boundary for the requested species.

        Parameters
        ----------
        species: str
            The name of the species for which to calculate the energy angular
            distribution function
        boundary: str
            The boundary at which to calculate the energy angular distribution
            function, one of 'z_lo', 'z_hi'

        Returns
        -------
        hist: np.ndarray
            The histogram of the energy angular distribution function
        '''
        # Get the wall eadf on the processor
        if species in self.species_names[1:]:
            hist = self._get_wall_ieadf(species, boundary)
        else:
            hist = self._get_wall_eeadf(species, boundary)

        # Sum histograms from all processors
        hist_all = np.zeros_like(hist)
        comm.Allreduce(hist, hist_all, op=mpi.SUM)
        self.wall_eadf_by_species[species][boundary] = hist_all

    def _get_wall_eeadf(self, species, boundary):
        '''
        Gets energy angular distribution functions organized into eeadf bins.
        '''
        if boundary not in ['z_lo', 'z_hi']:
            raise ValueError("Boundary must be one of 'z_lo' or 'z_hi'")

        # Set up wrappers
        boundary_wrapper = particle_containers.ParticleBoundaryBufferWrapper()

        try:
            ux = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary, 'ux', 0))
            uy = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary, 'uy', 0))
            uz = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary, 'uz', 0))
            w  = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary,  'w', 0))
        except ValueError:
            # Here if there are no ions at the boundary from this processor
            return np.zeros((len(self.eeadf_bin_centers), len(self.eadf_bin_centers)))

        # Calculate the electron energy and base its sign on the z velocity, but if z velocity is zero, use x velocity
        v2 = (np.square(ux) + np.square(uy) + np.square(uz))
        E = np.multiply(v2, 0.5 * self.mass_by_name[species] / constants.q_e)

        # Calculate the electron xy velocity
        vxy = np.sqrt(np.square(ux) + np.square(uy))
        # Calculate angle with a negative sign so that left/right wall eeadfs are on the left/right of an energy vs angle plot
        angle = np.arctan(vxy / uz) * 180 / np.pi

        # Get the histogram (unnormalized)
        hist, *_ = np.histogram2d(E, angle, bins=[self.eeadf_bin_edges, self.eadf_bin_edges], density=False, weights=w/self.dz)

        # hist = np.ascontiguousarray(hist, dtype=np.float64)
        hist = np.copy(hist, order='C')

        return hist

    def _get_wall_ieadf(self, species, boundary):
        '''
        Gets energy angular distribution functions organized into ieadf bins.
        '''
        if boundary not in ['z_lo', 'z_hi']:
            raise ValueError("Boundary must be one of 'z_lo' or 'z_hi'")

        # Set up wrappers
        boundary_wrapper = particle_containers.ParticleBoundaryBufferWrapper()

        try:
            ux = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary, 'ux', 0))
            uy = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary, 'uy', 0))
            uz = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary, 'uz', 0))
            w  = np.concatenate(boundary_wrapper.get_particle_boundary_buffer(species, boundary,  'w', 0))
        except ValueError:
            # Here if there are no ions at the boundary from this processor
            return np.zeros((len(self.ieadf_bin_centers), len(self.iadf_bin_centers)))

        # Calculate the ion energy and base its sign on the z velocity, but if z velocity is zero, use x velocity
        v2 = (np.square(ux) + np.square(uy) + np.square(uz))
        E = np.multiply(v2, 0.5 * self.mass_by_name[species] / constants.q_e)

        # Calculate the ion xy velocity
        vxy = np.sqrt(np.square(ux) + np.square(uy))
        # Calculate angle with a negative sign so that left/right wall ieadfs are on the left/right of an energy vs angle plot
        angle = np.arctan(vxy / uz) * 180 / np.pi

        # Get the histogram (unnormalized)
        hist, *_ = np.histogram2d(E, angle, bins=[self.ieadf_bin_edges, self.iadf_bin_edges], density=False, weights=w/self.dz)

        # hist = np.ascontiguousarray(hist, dtype=np.float64)
        hist = np.copy(hist, order='C')

        return hist

    def clear_wall_eadf_buffers(self):
        '''
        Clears the buffers for the energy angular distribution function.
        '''
        # Clear the boundary buffers
        boundary_wrapper = particle_containers.ParticleBoundaryBufferWrapper()
        boundary_wrapper.clear_buffer()

    ###########################################################################
    # Simulation Functions                                                    #
    ###########################################################################
    def do_diagnostics(self):
        '''
        Master function to perform diagnostics at each time step. Should be
        installed at least one step before the first diagnostic step.
        '''
        # leave if we are beyond a diagnostic collection
        if self.curr_diag_output >= self.num_outputs:
            return

        # Get current step
        step = self.sim_ext.warpx.getistep(lev=0)
        next_step = step + 1

        # Check if we are at the start of a new diagnostic output. We let the step prior enter,
        # so that we can save the electric field at the last step for the displacement current
        if next_step < self.diag_start[self.curr_diag_output]:
            return
        elif step == self.diag_start[self.curr_diag_output]:
            # Clear the wall eadf buffers for this collection
            self.clear_wall_eadf_buffers()

            # Clear WarpX's power deposition tracking buffers so the first
            # time-averaged collection of this diagnostic period isn't
            # contaminated by power accumulated before diag_start (e.g. during
            # a gap between diagnostic periods).
            ta_settings = self.master_diagnostic_dict['time_averaged']
            for species in self.species_names:
                if ta_settings.get(f'P_C_{species}', False) or ta_settings.get(f'P_I_{species}', False):
                    self.power_wrapper.clear_buffers([species], level=0)

        # Check if we need to save the electric field for the displacement current
        save_E_last_step = False
        if self.master_diagnostic_dict['time_resolved'].get('J_d', False) and (next_step - self.diag_start[self.curr_diag_output]) % self.diag_time_resolving_steps == 0:
            save_E_last_step = True
        if self.master_diagnostic_dict['time_averaged'].get('J_d', False) and (next_step - self.diag_start[self.curr_diag_output]) % self.diag_time_averaging_steps == 0:
            save_E_last_step = True
        if self.step_to_interval_map[self.curr_diag_output]:
            # Check if next_step is in the step_to_interval_map
            if next_step in self.step_to_interval_map[self.curr_diag_output] and self.master_diagnostic_dict['interval'].get('J_d', False):
                save_E_last_step = True

        # Go through each diagnostic type and determine if we need to update
        # arrays for that diagnostic at this time step
        time_resolved = False
        time_averaged = False
        interval = False
        if any(self.master_diagnostic_dict['time_resolved'].values()) and step >= self.diag_start[self.curr_diag_output] and ((step - self.diag_start[self.curr_diag_output]) % self.diag_time_resolving_steps == 0):
            time_resolved = True
        if any(self.master_diagnostic_dict['time_averaged'].values()) and step >= self.diag_start[self.curr_diag_output] and ((step - self.diag_start[self.curr_diag_output]) % self.diag_time_averaging_steps == 0):
            time_averaged = True
        if any(self.master_diagnostic_dict['interval'].values()) and self.step_to_interval_map[self.curr_diag_output]:
            # Fast lookup: check if current step is in the step_to_interval_map
            if step in self.step_to_interval_map[self.curr_diag_output]:
                interval = True
                # The current_interval_key is a tuple (interval_idx, slice_idx)
                # Extract the slice_idx for the diagnostic collection, while ignoring the interval_idx (not using at this time)
                _, slice_idx = self.step_to_interval_map[self.curr_diag_output][step]

        # Presave the electric field, if needed
        if any(
            d.get('E_z', False) or d.get('J_d', False) or any(k.startswith('P_C') and v for k, v in d.items())
            for d in self.master_diagnostic_dict.values()
        ) or save_E_last_step:
            np.copyto(self._current_Ez_data, self._Ez_wrapper[...])
        if any(
            d.get('E_y', False) or any(k.startswith('P_I') and v for k, v in d.items())
            for d in self.master_diagnostic_dict.values()
        ):
            np.copyto(self._current_Ey_data, self._Ey_wrapper[...])

        # Update arrays for diagnostics
        # Save which fields need to be updated this timestep
        active_diags = []
        if time_averaged:
            active_diags.append(self.master_diagnostic_dict['time_averaged'])
        if interval:
            active_diags.append(self.master_diagnostic_dict['interval'])
        if time_resolved:
            active_diags.append(self.master_diagnostic_dict['time_resolved'])

        diags_this_step = set()
        need_synchronization = False
        for mode_dict in active_diags:
            for key, value in mode_dict.items():
                if value:
                    diags_this_step.add(key)
                    need_synchronization |= key.startswith(self.VELOCITY_SYNC_PREFIXES)

        # Power diagnostics must use v^{n+1/2} paired with E^n (beforeEsolve state).
        # Run them before velocity synchronization to avoid the +dt/2 bias.
        for species in self.species_names:
            for prefix, func in self.SPECIES_DISPATCH.items():
                if prefix in self.POWER_PREFIXES:
                    key = f'{prefix}{species}'
                    if key in diags_this_step:
                        func(species)

        # Synchronize, if necessary, to catch velocities up to positions
        if need_synchronization:
            self.sim_ext.warpx.synchronize_velocity_with_position()

        # Call field diagnostics
        for diag, func in self.FIELD_DISPATCH.items():
            if diag in diags_this_step:
                func()

        # Call particle diagnostics (power already computed above)
        for species in self.species_names:
            for prefix, func in self.SPECIES_DISPATCH.items():
                if prefix not in self.POWER_PREFIXES:
                    key = f'{prefix}{species}'
                    if key in diags_this_step:
                        func(species)

        # Save diagnostics to arrays
        if time_resolved:
            self.do_time_resolved_diagnostics(self.curr_tr)
            self.curr_tr += 1
        if time_averaged:
            self.do_time_averaged_diagnostics()
        if interval:
            self.do_interval_diagnostics(slice_idx)

        # Save the electric field for the displacement current
        if save_E_last_step:
            np.copyto(self.E_z_last_step, self._current_Ez_data)

        # Finalize and save diagnostics
        if step == self.diag_stop[self.curr_diag_output]:

            # Save ieadf for each species and wall, if necessary
            if any(self.master_diagnostic_dict['ieadfs'].values()):
                for species in self.species_names[1:]:
                    for key, value in self.master_diagnostic_dict['ieadfs'].items():
                        if value:
                            self.calculate_wall_eadf(species, key)

            # Save eeadf, if necessary
            if any(self.master_diagnostic_dict['eeadfs'].values()):
                for key, value in self.master_diagnostic_dict['eeadfs'].items():
                    if value:
                        self.calculate_wall_eadf(self.electron_name, key)

            # Clear wall eadf buffers
            self.clear_wall_eadf_buffers()

            # Finalize and save diagnostic data
            self.save_diagnostic_data()

            # Move to next diagnostic output
            self.curr_diag_output += 1
            self.curr_tr = 0
            self.curr_interval = 0
            self.curr_slice = 0

            # Reset diagnostic arrays
            if self.curr_diag_output < self.num_outputs:
                self.reset_diagnostic_arrays()

    def do_time_resolved_diagnostics(self, tr_idx: int):
        '''
        Performs time resolved diagnostics

        Parameters
        ----------
        tr_idx: int
            Index of the time resolved diagnostic
        '''
        # Grab temporary dictionary for time resolved diagnostics
        temp_settings = self.master_diagnostic_dict['time_resolved']

        # Particle diagnostics
        for species in self.species_names:
            if temp_settings.get(f'N_{species}', False):
                self.tr_N[species][tr_idx] = self.N[species]
            if temp_settings.get(f'W_{species}', False):
                self.tr_W[species][tr_idx] = self.W[species]
            for dir in ('x', 'y', 'z'):
                if temp_settings.get(f'W{dir}_{species}', False):
                    getattr(self, f'tr_W{dir}')[species][tr_idx] = getattr(self, f'W{dir}')[species]
            for dir in ('z', 'y', 'x'):
                if temp_settings.get(f'J{dir}_{species}', False):
                    getattr(self, f'tr_J{dir}')[species][tr_idx] = getattr(self, f'J{dir}')[species]
            if temp_settings.get(f'P_C_{species}', False):
                self.tr_P_C[species][tr_idx] = self.P_C[species]
            if temp_settings.get(f'P_I_{species}', False):
                self.tr_P_I[species][tr_idx] = self.P_I[species]
            if temp_settings.get(f'Pw_{species}', False):
                self.tr_Pw[species][tr_idx] = self.Pw[species]
            if temp_settings.get(f'EDF_{species}', False):
                self.tr_EDF[species][tr_idx] = self.Edf[species]
            for evdf_prefix in self.EVDF_PREFIXES:
                key = f'{evdf_prefix}_{species}'
                if temp_settings.get(key, False):
                    self.tr_EVDF[key][tr_idx] += getattr(self, f'{evdf_prefix[:2]}df')[species]

        # Field diagnostics
        for dir in ('z', 'y', 'x'):
            if temp_settings.get(f'E_{dir}', False):
                self.tr_E[dir][tr_idx] = self.E[dir]
        if temp_settings.get('phi', False):
            self.tr_phi[tr_idx] = self.phi
        if temp_settings.get('J_d', False):
            self.tr_J_d[tr_idx] = self.J_d
        if temp_settings.get('J_w', False):
            self.tr_J_w[tr_idx] = self.J_w

        # Add time to time array
        self.tr_times[tr_idx] = self.sim_ext.warpx.gett_new(lev=0)

    def do_time_averaged_diagnostics(self):
        '''
        Performs time averaged diagnostics
        '''
        # Grab temporary dictionary for time averaged diagnostics
        temp_settings = self.master_diagnostic_dict['time_averaged']

        # Particle diagnostics: Add values now, average later
        for species in self.species_names:
            if temp_settings.get(f'N_{species}', False):
                self.ta_N[species] += self.N[species]
            if temp_settings.get(f'W_{species}', False):
                self.ta_W[species] += self.W[species]
                self.ta_W_collection_mask[species] += self.W_collection_mask[species]
            for dir in ('x', 'y', 'z'):
                if temp_settings.get(f'W{dir}_{species}', False):
                    getattr(self, f'ta_W{dir}')[species] += getattr(self, f'W{dir}')[species]
                    getattr(self, f'ta_W{dir}_collection_mask')[species] += getattr(self, f'W{dir}_collection_mask')[species]
            for dir in ('z', 'y', 'x'):
                if temp_settings.get(f'J{dir}_{species}', False):
                    getattr(self, f'ta_J{dir}')[species] += getattr(self, f'J{dir}')[species]
            if temp_settings.get(f'P_C_{species}', False):
                self.ta_P_C[species] += self._get_time_averaged_power_from_buffer(species, 'z')
            if temp_settings.get(f'P_I_{species}', False):
                self.ta_P_I[species] += self._get_time_averaged_power_from_buffer(species, 'y')
            if temp_settings.get(f'Pw_{species}', False):
                self.ta_Pw[species] += self.Pw[species]
            if temp_settings.get(f'EDF_{species}', False):
                self.ta_EDF[species] += self.Edf[species]
            for evdf_prefix in self.EVDF_PREFIXES:
                key = f'{evdf_prefix}_{species}'
                if temp_settings.get(key, False):
                    self.ta_EVDF[key] += getattr(self, f'{evdf_prefix[:2]}df')[species]

        # Field diagnostics
        for dir in ('z', 'y', 'x'):
            if temp_settings.get(f'E_{dir}', False):
                self.ta_E[dir] += self.E[dir]
        if temp_settings.get('phi', False):
            self.ta_phi += self.phi
        if temp_settings.get('J_d', False):
            self.ta_J_d += self.J_d
        if temp_settings.get('J_w', False):
            self.ta_J_w += self.J_w

    def do_interval_diagnostics(self, interval_idx: int):
        '''
        Perform diagnostics at an time within interval self.interval_time

        Parameters
        ----------
        interval_idx: int
            Index of interval in self.times_in_interval. Determines which array
            to update.
        '''
        # Grab temporary dictionary for interval diagnostics
        temp_settings = self.master_diagnostic_dict['interval']

        # Particle diagnostics: Add values now, average later
        for species in self.species_names:
            if temp_settings.get(f'N_{species}', False):
                self.in_N[species][interval_idx] += self.N[species]
            if temp_settings.get(f'W_{species}', False):
                self.in_W[species][interval_idx] += self.W[species]
                self.in_W_collection_mask[species][interval_idx] += self.W_collection_mask[species]
            for dir in ('x', 'y', 'z'):
                if temp_settings.get(f'W{dir}_{species}', False):
                    getattr(self, f'in_W{dir}')[species][interval_idx] += getattr(self, f'W{dir}')[species]
                    getattr(self, f'in_W{dir}_collection_mask')[species][interval_idx] += getattr(self, f'W{dir}_collection_mask')[species]
            for dir in ('z', 'y', 'x'):
                if temp_settings.get(f'J{dir}_{species}', False):
                    getattr(self, f'in_J{dir}')[species][interval_idx] += getattr(self, f'J{dir}')[species]
            if temp_settings.get(f'P_C_{species}', False):
                self.in_P_C[species][interval_idx] += self.P_C[species]
            if temp_settings.get(f'P_I_{species}', False):
                self.in_P_I[species][interval_idx] += self.P_I[species]
            if temp_settings.get(f'Pw_{species}', False):
                self.in_Pw[species][interval_idx] += self.Pw[species]
            if temp_settings.get(f'EDF_{species}', False):
                self.in_EDF[species][interval_idx] += self.Edf[species]
            for evdf_prefix in self.EVDF_PREFIXES:
                key = f'{evdf_prefix}_{species}'
                if temp_settings.get(key, False):
                    self.in_EVDF[key][interval_idx] += getattr(self, f'{evdf_prefix[:2]}df')[species]

        # Field diagnostics
        for dir in ('z', 'y', 'x'):
            if temp_settings.get(f'E_{dir}', False):
                self.in_E[dir][interval_idx] += self.E[dir]
        if temp_settings.get('phi', False):
            self.in_phi[interval_idx] += self.phi
        if temp_settings.get('J_d', False):
            self.in_J_d[interval_idx] += self.J_d
        if temp_settings.get('J_w', False):
            self.in_J_w[interval_idx] += self.J_w

    def reset_diagnostic_arrays(self):
        '''
        Reset diagnostic arrays, call after the diagnostic output
        counter has been incremented.
        '''
        # Ieadf arrays
        self.wall_eadf_by_species = {}
        for species in self.species_names[1:]:
            self.wall_eadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['ieadfs'].items():
                if value:
                    self.wall_eadf_by_species[species][key] = np.zeros((len(self.ieadf_bin_centers), len(self.iadf_bin_centers)))

        # Eeadf arrays
        self.wall_eadf_by_species[self.electron_name] = {}
        for key, value in self.master_diagnostic_dict['eeadfs'].items():
            if value:
                self.wall_eadf_by_species[self.electron_name][key] = np.zeros((len(self.eeadf_bin_centers), len(self.eadf_bin_centers)))

        # Time resolved arrays - dictionary-based storage by species name
        self.tr_N = {key.replace('N_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                     for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('N_')}
        self.tr_W = {key.replace('W_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                     for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('W_')}
        self.tr_Wx = {key.replace('Wx_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Wx_')}
        self.tr_Wy = {key.replace('Wy_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Wy_')}
        self.tr_Wz = {key.replace('Wz_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Wz_')}
        self.tr_Jz = {key.replace('Jz_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Jz_')}
        self.tr_Jy = {key.replace('Jy_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Jy_')}
        self.tr_Jx = {key.replace('Jx_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Jx_')}
        self.tr_P_C = {key.replace('P_C_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('P_C_')}
        self.tr_P_I = {key.replace('P_I_', ''): np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('P_I_')}
        self.tr_Pw = {key.replace('Pw_', ''): np.zeros((self.tr_coll[self.curr_diag_output], 2))
                      for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('Pw_')}

        # Field diagnostics (not species-specific)
        self.tr_E = {
            'z': np.zeros((self.tr_coll[self.curr_diag_output], self.nz)),
            'y': np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1)),
            'x': np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        }
        self.tr_phi = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_J_d = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_J_w = np.zeros((self.tr_coll[self.curr_diag_output], 2))

        # Distribution functions by species
        self.tr_EDF = {key.replace('EDF_', ''): np.zeros((self.tr_coll[self.curr_diag_output], len(self.edf_bounds) + 1, len(self.edf_centers_by_species[key.replace('EDF_', '')])))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('EDF_')}
        self.tr_EVDF = {key: np.zeros((self.tr_coll[self.curr_diag_output], len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                       for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('ExDF_')}
        self.tr_EVDF.update({key: np.zeros((self.tr_coll[self.curr_diag_output], len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('EyDF_')})
        self.tr_EVDF.update({key: np.zeros((self.tr_coll[self.curr_diag_output], len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_resolved'] if key.startswith('EzDF_')})

        self.tr_times = np.zeros((self.tr_coll[self.curr_diag_output]))

        # Power arrays
        self.tr_Pin_vst = None

        # Time averaged arrays - dictionary-based storage by species name
        self.ta_N = {key.replace('N_', ''): np.zeros(self.nz + 1)
                     for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('N_')}
        self.ta_W = {key.replace('W_', ''): np.zeros(self.nz + 1)
                     for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('W_')}
        self.ta_W_collection_mask = {key.replace('W_', ''): np.zeros(self.nz + 1)
                                     for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('W_')}
        self.ta_Wx = {key.replace('Wx_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wx_')}
        self.ta_Wx_collection_mask = {key.replace('Wx_', ''): np.zeros(self.nz + 1)
                                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wx_')}
        self.ta_Wy = {key.replace('Wy_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wy_')}
        self.ta_Wy_collection_mask = {key.replace('Wy_', ''): np.zeros(self.nz + 1)
                                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wy_')}
        self.ta_Wz = {key.replace('Wz_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wz_')}
        self.ta_Wz_collection_mask = {key.replace('Wz_', ''): np.zeros(self.nz + 1)
                                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Wz_')}
        self.ta_Jz = {key.replace('Jz_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Jz_')}
        self.ta_Jy = {key.replace('Jy_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Jy_')}
        self.ta_Jx = {key.replace('Jx_', ''): np.zeros(self.nz + 1)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Jx_')}
        self.ta_P_C = {key.replace('P_C_', ''): np.zeros(self.nz)
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('P_C_')}
        self.ta_P_I = {key.replace('P_I_', ''): np.zeros(self.nz)
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('P_I_')}
        self.ta_Pw = {key.replace('Pw_', ''): np.zeros(2)
                      for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('Pw_')}

        # Field diagnostics (not species-specific)
        self.ta_E = {
            'z': np.zeros(self.nz),
            'y': np.zeros(self.nz + 1),
            'x': np.zeros(self.nz + 1)
        }
        self.ta_phi = np.zeros(self.nz + 1)
        self.ta_J_d = np.zeros(self.nz)
        self.ta_J_w = np.zeros(2)

        # Distribution functions
        self.ta_EDF = {key.replace('EDF_', ''): np.zeros((len(self.edf_bounds) + 1, len(self.edf_centers_by_species[key.replace('EDF_', '')])))
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('EDF_')}
        self.ta_EVDF = {key: np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                       for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('ExDF_')}
        self.ta_EVDF.update({key: np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('EyDF_')})
        self.ta_EVDF.update({key: np.zeros((len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['time_averaged'] if key.startswith('EzDF_')})

        # Interval arrays - dictionary-based storage by species name
        self.in_N = {key.replace('N_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                     for key in self.master_diagnostic_dict['interval'] if key.startswith('N_')}
        self.in_W = {key.replace('W_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                     for key in self.master_diagnostic_dict['interval'] if key.startswith('W_')}
        self.in_W_collection_mask = {key.replace('W_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                     for key in self.master_diagnostic_dict['interval'] if key.startswith('W_')}
        self.in_Wx = {key.replace('Wx_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wx_')}
        self.in_Wx_collection_mask = {key.replace('Wx_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wx_')}
        self.in_Wy = {key.replace('Wy_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wy_')}
        self.in_Wy_collection_mask = {key.replace('Wy_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wy_')}
        self.in_Wz = {key.replace('Wz_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wz_')}
        self.in_Wz_collection_mask = {key.replace('Wz_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Wz_')}
        self.in_Jz = {key.replace('Jz_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Jz_')}
        self.in_Jy = {key.replace('Jy_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Jy_')}
        self.in_Jx = {key.replace('Jx_', ''): np.zeros((len(self.in_slices), self.nz + 1))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Jx_')}
        self.in_P_C = {key.replace('P_C_', ''): np.zeros((len(self.in_slices), self.nz))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('P_C_')}
        self.in_P_I = {key.replace('P_I_', ''): np.zeros((len(self.in_slices), self.nz))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('P_I_')}
        self.in_Pw = {key.replace('Pw_', ''): np.zeros((len(self.in_slices), 2))
                      for key in self.master_diagnostic_dict['interval'] if key.startswith('Pw_')}

        # Field diagnostics (not species-specific)
        self.in_E = {
            'z': np.zeros((len(self.in_slices), self.nz)),
            'y': np.zeros((len(self.in_slices), self.nz + 1)),
            'x': np.zeros((len(self.in_slices), self.nz + 1))
        }
        self.in_phi = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_J_d = np.zeros((len(self.in_slices), self.nz))
        self.in_J_w = np.zeros((len(self.in_slices), 2))

        # Distribution functions
        self.in_EDF = {key.replace('EDF_', ''): np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.edf_centers_by_species[key.replace('EDF_', '')])))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('EDF_')}
        self.in_EVDF = {key: np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                       for key in self.master_diagnostic_dict['interval'] if key.startswith('ExDF_')}
        self.in_EVDF.update({key: np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['interval'] if key.startswith('EyDF_')})
        self.in_EVDF.update({key: np.zeros((len(self.in_slices), len(self.edf_bounds) + 1, len(self.evdf_centers_by_diag_name[key])))
                            for key in self.master_diagnostic_dict['interval'] if key.startswith('EzDF_')})

        # Collision trackers
        active_coll_trackers = [key for key in self.master_diagnostic_dict['collisional'] if self.master_diagnostic_dict['collisional'][key]]
        self.collision_wrapper.clear_buffers(active_coll_trackers)

    ###########################################################################
    # Saving Functions                                                        #
    ###########################################################################
    def _finalize_diagnostic_data(self):
        '''
        Finalize diagnostic data before saving
        '''
        # -------------------------------------------------------
        # Grab temporary dictionary for time resolved diagnostics
        active = self.master_diagnostic_dict['time_resolved']

        # Convert to correct units
        for key in active:
            if not active.get(key, False):
                continue

            if any(key.startswith(prefix) for prefix in self.PARTICLE_DIAGNOSTIC_PREFIXES):
                prefix = '_'.join(key.split('_')[:-1])
                species = key.split('_')[-1]

                if prefix == 'N':
                    self.tr_N[species] /= self.dz
                if prefix == 'W':
                    self.tr_W[species] *= self.mass_by_name[species] / 2.0 / constants.q_e
                if prefix in ('Wx', 'Wy', 'Wz'):
                    getattr(self, f'tr_{prefix}')[species] *= self.mass_by_name[species] / 2.0 / constants.q_e
                if prefix in ('Jz', 'Jy', 'Jx'):
                    getattr(self, f'tr_{prefix}')[species] *= self.charge_by_name[species] / self.dz
                if prefix == 'P_C':
                    self.tr_P_C[species] *= self.charge_by_name[species] / self.dz
                if prefix == 'P_I':
                    self.tr_P_I[species] *= self.charge_by_name[species] / self.dz
                if prefix == 'Pw':
                    self.tr_Pw[species] /= self.dt

            else:
                if key == 'J_d':
                    self.tr_J_d *= constants.ep0 / self.dt
                if key == 'J_w':
                    self.tr_J_w *= constants.q_e / self.dt

        # Calculate power input diagnostic
        if self.tr_power_dict.get('Pin_vst', False):
            self.tr_Pin_vst = np.zeros(len(self.tr_times))
            for time_idx in range(len(self.tr_times)):
                # Sum currents from all species at boundary
                total_Jz_boundary = sum(self.tr_Jz[species][time_idx][-1] for species in self.tr_Jz.keys())
                self.tr_Pin_vst[time_idx] = -total_Jz_boundary * self.tr_phi[time_idx][-1]

        # -------------------------------------------------------
        # Grab temporary dictionary for time averaged diagnostics
        active = self.master_diagnostic_dict['time_averaged']
        collections = self.ta_coll[self.curr_diag_output]

        # Convert to correct units
        for key in active:
            if not active.get(key, False):
                continue

            if any(key.startswith(prefix) for prefix in self.PARTICLE_DIAGNOSTIC_PREFIXES):
                prefix = '_'.join(key.split('_')[:-1])
                species = key.split('_')[-1]

                if prefix == 'N':
                    self.ta_N[species] /= collections * self.dz
                if prefix == 'W':
                    v2_factor = self.mass_by_name[species] / 2.0 / constants.q_e
                    self.ta_W[species] = np.divide(self.ta_W[species] * v2_factor, self.ta_W_collection_mask[species],
                                                   out=np.zeros_like(self.ta_W[species]),
                                                   where=self.ta_W_collection_mask[species]!=0)
                if prefix in ('Wx', 'Wy', 'Wz'):
                    v2_factor = self.mass_by_name[species] / 2.0 / constants.q_e
                    ta_dict = getattr(self, f'ta_{prefix}')
                    ta_mask = getattr(self, f'ta_{prefix}_collection_mask')
                    ta_dict[species] = np.divide(ta_dict[species] * v2_factor, ta_mask[species],
                                                 out=np.zeros_like(ta_dict[species]),
                                                 where=ta_mask[species]!=0)
                if prefix in ('Jz', 'Jy', 'Jx'):
                    getattr(self, f'ta_{prefix}')[species] *= self.charge_by_name[species] / self.dz / collections
                # P_C/P_I (time-averaged) are read from WarpX's power deposition
                # tracking buffer, which already includes charge and is divided
                # by cell size and elapsed steps at each collection (see
                # _get_time_averaged_power_from_buffer). Only the average over
                # collection windows within this diagnostic period remains.
                if prefix == 'P_C':
                    self.ta_P_C[species] /= collections
                if prefix == 'P_I':
                    self.ta_P_I[species] /= collections
                if prefix == 'Pw':
                    self.ta_Pw[species] /= self.dt * collections
                if prefix == 'EDF':
                    self.ta_EDF[species] /= collections
                if prefix in self.EVDF_PREFIXES:
                    self.ta_EVDF[f'{prefix}_{species}'] /= collections

            else:
                for dir in ('z', 'y', 'x'):
                    if key == f'E_{dir}':
                        self.ta_E[dir] /= collections
                if key == 'phi':
                    self.ta_phi /= collections
                if key == 'J_d':
                    self.ta_J_d *= constants.ep0 / self.dt / collections
                if key == 'J_w':
                    self.ta_J_w *= constants.q_e / self.dt / collections

        # --------------------------------------------------
        # Grab temporary dictionary for interval diagnostics
        active = self.master_diagnostic_dict['interval']

        # Create a dictionary to count collections for each slice
        collection_counts = {}
        for ii in range(len(self.in_slices)):
            collection_counts[ii] = 0

        # Count the number of collections for each slice
        for (_, slice_idx), count in self.in_coll_counts[self.curr_diag_output].items():
            collection_counts[slice_idx] += count

        # Convert to correct units
        for key in active:
            if not active.get(key, False):
                continue

            if any(key.startswith(prefix) for prefix in self.PARTICLE_DIAGNOSTIC_PREFIXES):
                prefix = '_'.join(key.split('_')[:-1])
                species = key.split('_')[-1]

                if prefix == 'N':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_N[species][ii] /= collection_counts[ii] * self.dz
                if prefix == 'W':
                    v2_factor = self.mass_by_name[species] / 2.0 / constants.q_e
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_W[species][ii] = np.divide(self.in_W[species][ii] * v2_factor, self.in_W_collection_mask[species][ii],
                                                           out=np.zeros_like(self.in_W[species][ii]),
                                                           where=self.in_W_collection_mask[species][ii]!=0)
                if prefix in ('Wx', 'Wy', 'Wz'):
                    v2_factor = self.mass_by_name[species] / 2.0 / constants.q_e
                    in_dict = getattr(self, f'in_{prefix}')
                    in_mask = getattr(self, f'in_{prefix}_collection_mask')
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        in_dict[species][ii] = np.divide(in_dict[species][ii] * v2_factor, in_mask[species][ii],
                                                         out=np.zeros_like(in_dict[species][ii]),
                                                         where=in_mask[species][ii]!=0)
                if prefix in ('Jz', 'Jy', 'Jx'):
                    in_dict = getattr(self, f'in_{prefix}')
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        in_dict[species][ii] *= self.charge_by_name[species] / self.dz / collection_counts[ii]
                if prefix == 'P_C':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_P_C[species][ii] *= self.charge_by_name[species] / self.dz / collection_counts[ii]
                if prefix == 'P_I':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_P_I[species][ii] *= self.charge_by_name[species] / self.dz / collection_counts[ii]
                if prefix == 'Pw':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_Pw[species][ii] /= self.dt * collection_counts[ii]
                if prefix == 'EDF':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_EDF[species][ii] /= collection_counts[ii]
                if prefix in self.EVDF_PREFIXES:
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_EVDF[f'{prefix}_{species}'][ii] /= collection_counts[ii]

            else:
                for dir in ('z', 'y', 'x'):
                    if key == f'E_{dir}':
                        for ii in range(len(self.in_slices)):
                            if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                                continue
                            self.in_E[dir][ii] /= collection_counts[ii]
                if key == 'phi':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_phi[ii] /= collection_counts[ii]
                if key == 'J_d':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_J_d[ii] *= constants.ep0 / self.dt / collection_counts[ii]
                if key == 'J_w':
                    for ii in range(len(self.in_slices)):
                        if not self.in_coll_steps[self.curr_diag_output] or collection_counts[ii] == 0:
                            continue
                        self.in_J_w[ii] *= constants.q_e / self.dt / collection_counts[ii]

    def save_diagnostic_data(self):
        '''
        Save diagnostic data at the current time step
        '''
        if any(self.master_diagnostic_dict['collisional'].values()):
            coll_data = {}
            active = self.master_diagnostic_dict['collisional']
            for key in active:
                if active.get(key, False):
                    coll_data[key] = self.collision_wrapper.get_all(
                        key, level=0, copy_to_host=True, energy_units='J'
                    )

        if comm.rank != 0:
            return

        self._finalize_diagnostic_data()

        # If we are saving anywhere other than the end step of a diagnostic
        # output, we save the file as if at diagnostic output -1. This avoids overwriting.
        if self.sim_ext.warpx.getistep(lev=0) not in self.diag_stop:
            step = -1
        else:
            step = self.curr_diag_output + 1

        # Make sure the directory exists
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)

        # Create directories for each diagnostic type
        tr_folder = os.path.join(self.diag_folder, f'time_resolved_{step:04d}')
        ta_folder = os.path.join(self.diag_folder, f'time_averaged_{step:04d}')
        in_folder = os.path.join(self.diag_folder, f'interval_{step:04d}')
        if any(self.master_diagnostic_dict['time_resolved'].values()) and not os.path.exists(tr_folder):
            os.makedirs(tr_folder)
        if (any(self.master_diagnostic_dict['time_averaged'].values()) or any(self.master_diagnostic_dict['collisional'].values())) and not os.path.exists(ta_folder):
            os.makedirs(ta_folder)
        if any(self.master_diagnostic_dict['interval'].values()) and not os.path.exists(in_folder):
            os.makedirs(in_folder)

        # Save collision trackers
        active = self.master_diagnostic_dict['collisional']
        if any(active.values()):
            coll_collection_time = self._get_collision_accumulation_time(self.curr_diag_output)
        for key in active:
            if active.get(key, False):
                if coll_data[key] is None:
                    print(f"Warning: No collision data available for {key} at step {step}. Skipping saving for this diagnostic.")
                    continue
                for process_name in coll_data[key]:
                    filename = os.path.join(ta_folder, f'coll-rate_{key}_{process_name}.npy')
                    rate = coll_data[key][process_name][0] / coll_collection_time / self.dz
                    np.save(filename, rate)

                    filename = os.path.join(ta_folder, f'coll-energy_{key}_{process_name}.npy')
                    en_rate = coll_data[key][process_name][1] / coll_collection_time / self.dz
                    np.save(filename, en_rate)

        # Save ieadfs
        active = self.master_diagnostic_dict['ieadfs']
        for key, val in active.items():
            if val:
                for species in self.species_names[1:]:
                    if key == 'z_lo':
                        prefix = 'lw'
                    elif key == 'z_hi':
                        prefix = 'rw'
                    np.save(os.path.join(self.wall_eadf_dir_by_species[species], f'{prefix}_{step:04d}.npy'), self.wall_eadf_by_species[species][key])

        # Save eeadfs
        active = self.master_diagnostic_dict['eeadfs']
        for key, val in active.items():
            if val:
                if key == 'z_lo':
                    prefix = 'lw'
                elif key == 'z_hi':
                    prefix = 'rw'
                np.save(os.path.join(self.wall_eadf_dir_by_species[self.electron_name], f'{prefix}_{step:04d}.npy'), self.wall_eadf_by_species[self.electron_name][key])

        # Save time resolved diagnostics
        active = self.master_diagnostic_dict['time_resolved']
        for key in active:
            if not active.get(key, False):
                continue

            if key in self.FIELD_DIAGNOSTICS:
                filename = os.path.join(tr_folder, f'{key}.npy')
                if key in ['E_z', 'E_y', 'E_x']:
                    diag_attr = getattr(self, f'tr_E')[key.split('_')[1]]
                else:
                    diag_attr = getattr(self, f'tr_{key}')
                np.save(filename, diag_attr)
                continue

            prefix = '_'.join(key.split('_')[:-1])
            species = key.split('_')[-1]

            # Handle distribution function diagnostics (named according to EDF_species)
            if prefix == 'EDF':
                edf_attr = getattr(self, 'tr_EDF')
                for ii in range(len(edf_attr[species][0])):
                    filename = os.path.join(tr_folder, f'{key}_{ii+1:02d}.npy')
                    np.save(filename, edf_attr[species][:,ii])
                continue
            if prefix in self.EVDF_PREFIXES:
                vdf_attr = getattr(self, 'tr_EVDF')
                for ii in range(len(vdf_attr[key][0])):
                    filename = os.path.join(tr_folder, f'{key}_{ii+1:02d}.npy')
                    np.save(filename, vdf_attr[key][:,ii])
                continue

            # Handle particle diagnostics
            filename = os.path.join(tr_folder, f'{key}.npy')
            diag_attr = getattr(self, f'tr_{prefix}')
            if isinstance(diag_attr, dict):
                np.save(filename, diag_attr[species])
            else:
                np.save(filename, diag_attr)

        if any(active.values()):
            np.save(os.path.join(tr_folder, 'times.npy'), self.tr_times)

        # Save time resolved power diagnostics
        for key in self.tr_power_dict:
            if self.tr_power_dict.get(key, False):
                np.save(os.path.join(tr_folder, f'{key}.npy'), getattr(self, f'tr_{key}'))

        # Save time averaged diagnostics
        active = self.master_diagnostic_dict['time_averaged']
        for key in active:
            if not active.get(key, False):
                continue

            if key in self.FIELD_DIAGNOSTICS:
                filename = os.path.join(ta_folder, f'{key}.npy')
                if key in ['E_z', 'E_y', 'E_x']:
                    diag_attr = getattr(self, f'ta_E')[key.split('_')[1]]
                else:
                    diag_attr = getattr(self, f'ta_{key}')
                np.save(filename, diag_attr)
                continue

            prefix = '_'.join(key.split('_')[:-1])
            species = key.split('_')[-1]

            # Handle distribution function diagnostics (named according to EDF_species)
            if prefix == 'EDF':
                edf_attr = getattr(self, 'ta_EDF')
                for ii in range(len(edf_attr[species])):
                    filename = os.path.join(ta_folder, f'{key}_{ii+1:02d}.npy')
                    np.save(filename, edf_attr[species][ii])
                continue
            if prefix in self.EVDF_PREFIXES:
                vdf_attr = getattr(self, 'ta_EVDF')
                for ii in range(len(vdf_attr[key])):
                    filename = os.path.join(ta_folder, f'{key}_{ii+1:02d}.npy')
                    np.save(filename, vdf_attr[key][ii])
                continue

            # Handle particle diagnostics
            filename = os.path.join(ta_folder, f'{key}.npy')
            diag_attr = getattr(self, f'ta_{prefix}')
            if isinstance(diag_attr, dict):
                np.save(filename, diag_attr[species])
            else:
                np.save(filename, diag_attr)

        # Save interval diagnostics
        active = self.master_diagnostic_dict['interval']
        if len(self.in_coll_steps[self.curr_diag_output]) > 0:
            for key in active:
                if not active.get(key, False):
                    continue

                if key in self.FIELD_DIAGNOSTICS:
                    filename = os.path.join(in_folder, f'{key}.npz')
                    if key in ['E_z', 'E_y', 'E_x']:
                        diag_attr = getattr(self, f'in_E')[key.split('_')[1]]
                    else:
                        diag_attr = getattr(self, f'in_{key}')
                    arrays_dict = {f't{i+1:02d}': diag_attr[i] for i in range(len(self.in_slices))}
                    np.savez(filename, **arrays_dict)
                    continue

                prefix = '_'.join(key.split('_')[:-1])
                species = key.split('_')[-1]

                # Handle distribution function diagnostics (named according to EDF_species)
                if prefix == 'EDF':
                    edf_attr = getattr(self, 'in_EDF')
                    for ii in range(len(edf_attr[species][0])):
                        arrays_dict = {f't{i+1:02d}': edf_attr[species][i, ii] for i in range(len(self.in_slices))}
                        filename = os.path.join(in_folder, f'{key}_{ii+1:02d}.npz')
                        np.savez(filename, **arrays_dict)
                    continue
                if prefix in self.EVDF_PREFIXES:
                    vdf_attr = getattr(self, 'in_EVDF')
                    for ii in range(len(vdf_attr[key][0])):
                        arrays_dict = {f't{i+1:02d}': vdf_attr[key][i, ii] for i in range(len(self.in_slices))}
                        filename = os.path.join(in_folder, f'{key}_{ii+1:02d}.npz')
                        np.savez(filename, **arrays_dict)
                    continue

                # Handle particle diagnostics
                filename = os.path.join(in_folder, f'{key}.npz')
                diag_attr = getattr(self, f'in_{prefix}')
                if isinstance(diag_attr, dict):
                    arrays_dict = {f't{i+1:02d}': diag_attr[species][i] for i in range(len(self.in_slices))}
                else:
                    arrays_dict = {f't{i+1:02d}': diag_attr[i] for i in range(len(self.in_slices))}
                np.savez(filename, **arrays_dict)

    ###########################################################################
    # Helper Functions                                                        #
    ###########################################################################
    def _check_file(self, file_name):
        '''
        If the file exists, rename it to have '_old' before the extension.
        '''
        if os.path.exists(file_name):
            # Split the file name at the '.' and add '_old' before the extension
            file_name_split = file_name.split('.')
            file_name_split[-2] += '_old'
            old_file_name = '.'.join(file_name_split)
            os.rename(file_name, old_file_name)

    def _get_collision_accumulation_time(self, diag_output: int) -> float:
        '''
        Get the total time over which collisions were accumulated for a given
        diagnostic output. Collisions are cleared at the beginning of the first
        collection and at the end of each diagnostic output.

        Parameters
        ----------
        diag_output: int
            Index of the diagnostic output.

        Returns
        -------
        float
            Total time [s] over which collisions were collected for
            the given diagnostic output
        '''
        if diag_output == 0:
            return self.diag_time + self.dt
        return self.diag_time + self.evolve_time
