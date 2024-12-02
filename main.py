from __future__ import annotations  # Allows using class names as type hints before they are fully defined
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inputs import CapacitiveDischargeExample  # Only imports Simulation for type checking to avoid circular import

import numpy as np
import sys, os

from pywarpx import fields, particle_containers, picmi
from mpi4py import MPI as mpi
import time

# Initialize mpi communicator
comm = mpi.COMM_WORLD
num_proc = comm.Get_size()

constants = picmi.constants

class ICPHeatSource:
    def __init__(self,
                 simulation_obj: CapacitiveDischargeExample,
                 sim_ext: picmi.Simulation.extension,
                 ion_spec_names: list = None,
                 diag_outfolder: str = './diags'
                 ):
        '''
        Class which inputs power into the plasma via an external ICP Field.
        The simulation class must have the following attributes:
        - ICP_Jmag: float
        - ICP_freq: float
        - ICP_zmin: float
        - ICP_zmax: float

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        sim_ext: picmi.Simulation.extension
            Simulation extension object
        ion_spec_names: list, optional
            List of ion species names
        diag_outfolder: str, optional
            Folder to save diagnostics
        '''
        # Import simulation extension object
        self.sim_ext = sim_ext

        # Import simulation parameters
        self.Jmag        = simulation_obj.ICP_Jmag
        self.freq        = simulation_obj.ICP_freq
        self.zmin        = simulation_obj.ICP_zmin
        self.zmax        = simulation_obj.ICP_zmax

        self.dt          = simulation_obj.dt
        self.dz          = simulation_obj.dz
        self.nz          = simulation_obj.nz

        # Import general timing parameters
        self._import_general_timing_info(simulation_obj)

        self.species_names = ['electrons']
        if ion_spec_names is not None:
            self.species_names += ion_spec_names

        self.charge_by_name = {
            'electrons': -constants.q_e,
            ion_spec_names[0]: constants.q_e
        }

        self.diag_outfolder = diag_outfolder

        # Initialize
        self._initialize_ICP(simulation_obj=simulation_obj)
        self._initialize_ICP_diagnostics()

    def _initialize_ICP(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Initialize the ICP region.

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        # Get the lower and upper bounds of the ICP region
        self.zmin_idx = int(self.zmin / self.dz)
        self.zmax_idx = int(self.zmax / self.dz)

        # Resave the zmin and zmax values
        self.zmin = self.zmin_idx * self.dz
        self.zmax = self.zmax_idx * self.dz

        # Calculate the coefficients for the push
        self.t_coeff = 2 * np.pi * self.freq
        self.E_field_coeff = self.dt / constants.ep0
        self.accel_coeff = {}
        self.accel_coeff['electrons'] = self.charge_by_name['electrons'] * self.dt / constants.m_e
        for species in self.species_names[1:]:
            self.accel_coeff[species] = self.charge_by_name[species] * self.dt / simulation_obj.m_ion

        # NEED TO CONFIRM ARRAYS ARE CORRECT EVERYWHERE
        self.ICP_window_size = self.zmax_idx - self.zmin_idx

        # Initialize the ICP field
        self.E_ICP = np.zeros(self.ICP_window_size + 1)

        # Save pertinant information to file './diags/ICP_info.dat'
        if comm.rank != 0:
            return
        
        # Make a diagnostics directory
        if not os.path.exists(self.diag_outfolder):
            os.makedirs(self.diag_outfolder)

        self.ICP_dir = os.path.join(self.diag_outfolder, 'ICP')
        if not os.path.exists(self.ICP_dir):
            os.makedirs(self.ICP_dir)

        file = os.path.join(self.ICP_dir, 'ICP_info.dat')
        with open(file, 'w') as f:
            f.write('ICP Parameters\n')
            f.write('--------------\n')
            f.write(f'Source [A/m^2]={self.Jmag}*sin(2*pi*{self.freq}*t)\n')
            f.write(f'zmin [m]={self.zmin:e}\n')
            f.write(f'zmax [m]={self.zmax:e}\n')
            f.write(f'zmin_idx={self.zmin_idx}\n')
            f.write(f'zmax_idx={self.zmax_idx}\n\n')

            f.write(f'ICP Window Size=zmax_idx-zmin_idx={self.ICP_window_size}\n')

            f.write('Simulation Parameters\n')
            f.write('---------------------\n')
            f.write(f'dt [s]={self.dt:e}\n')
            f.write(f'dz [m]={self.dz:e}\n')

    def _initialize_ICP_diagnostics(self):
        '''
        Initialize the ICP diagnostics.
        '''
        # Set the diagnostic output index
        self.curr_diag_output = 0

        # Get the number of steps within the first diagnostic collection
        self.diag_step_count = self.diag_stop[self.curr_diag_output] - self.diag_start[self.curr_diag_output]

        if comm.rank != 0:
            return

        # Initialize the ICP recording arrays
        self.Jperp_diag = np.zeros((self.diag_step_count + 1, self.ICP_window_size + 1))
        self.field_diag = np.zeros((self.diag_step_count + 1, self.ICP_window_size + 1))

    def _import_general_timing_info(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Import diagnostic steps for diagnostics.
        
        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        # Import simulation parameters
        self.diag_time         = simulation_obj.diag_time
        self.evolve_time       = simulation_obj.evolve_time

        self.num_outputs       = simulation_obj.num_diag_steps
        self.max_output_idx    = self.num_outputs - 1

        self.diag_start        = simulation_obj.diag_start
        self.diag_stop         = simulation_obj.diag_stop

    ###########################################################################
    # Functions                                                               #
    ###########################################################################
    def calculate_E_ICP(self):
        '''
        Function which heats particles in the plasma using an ICP field.
        '''
        # Synchronize positions and velocities
        self.sim_ext.warpx.synchronize()

        # Calculate perpendicular conduction current density
        J_conduction = self.calculate_J_perp()

        # Calculate the perscribed current density
        J_prescribed = self.Jmag * np.sin(self.t_coeff * self.sim_ext.warpx.gett_new(lev=0))

        # Calculate the difference between the two current densities
        J_diff = J_prescribed - J_conduction

        # Calculate the ICP field
        self.E_ICP += J_diff * self.E_field_coeff

        # Make an array of length nz + 1 and fill it with the ICP field
        E_ICP_full = np.zeros(self.nz + 1)

        # Collected on the nodes, so put on the nodes directly
        E_ICP_full[self.zmin_idx:self.zmax_idx+1] = self.E_ICP

        # Get the Ex field wrapper
        Ex = fields.ExFPWrapper()

        Ex[...] = E_ICP_full

        step = self.sim_ext.warpx.getistep(lev=0)

        # Save the current density and field to the diagnostic arrays
        if comm.rank != 0:
            return

        step = self.sim_ext.warpx.getistep(lev=0)
        if step > self.diag_stop[self.max_output_idx] or step < self.diag_start[self.curr_diag_output]:
            return

        coll_idx = step - self.diag_start[self.curr_diag_output]

        self.Jperp_diag[coll_idx] = J_conduction
        self.field_diag[coll_idx] = E_ICP_full[self.zmin_idx : self.zmax_idx + 1]

        if coll_idx == self.diag_step_count:
            self.curr_diag_output += 1
            self._save_data()

            if self.curr_diag_output != self.num_outputs:
                self.diag_step_count = self.diag_stop[self.curr_diag_output] - self.diag_start[self.curr_diag_output]
                self.Jperp_diag = np.zeros((self.diag_step_count + 1, self.ICP_window_size + 1))
                self.field_diag = np.zeros((self.diag_step_count + 1, self.ICP_window_size + 1))

    def calculate_J_perp(self):
        '''
        Calculate the perpendicular conduction current density.
        '''
        # Get the current density of all species
        J_perp = np.zeros(self.ICP_window_size + 1)
        for species in self.species_names:
            J_perp += self.charge_by_name[species] * self.calculate_Flux_perp(species)

        return J_perp

    def calculate_Flux_perp(self, species) -> np.ndarray[float]:
        '''
        Calculate the perpendicular flux density for a species.

        Parameters
        ----------
        species: str
            Name of species
        '''
        # Get wrappers for species
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle data
        try:
            ux = np.concatenate(species_wrapper.get_particle_ux())
            z = np.concatenate(species_wrapper.get_particle_z())
            w = np.concatenate(species_wrapper.get_particle_weight())
        except ValueError:
            ux = np.array([])
            z = np.array([])
            w = np.array([])

        Flux_perp_temp = np.zeros(self.ICP_window_size + 1)

        # Calculate cell indices and fractional positions
        cell_idx = np.floor_divide(z, self.dz).astype(int) - self.zmin_idx
        frac_pos = (z / self.dz) - (cell_idx + self.zmin_idx)

        # Create masks for boundary conditions
        mask_first = (cell_idx == 0)
        mask_last = (cell_idx == self.ICP_window_size - 1)
        mask_middle = (cell_idx > 0) & (cell_idx < self.ICP_window_size - 1)

        # First boundary (cell_idx == 0)
        if np.any(mask_first):  # Check if there are any True values in mask_first
            np.add.at(Flux_perp_temp, 0, np.sum(ux[mask_first] * w[mask_first] * (1 - frac_pos[mask_first]) * 2))
            np.add.at(Flux_perp_temp, 1, np.sum(ux[mask_first] * w[mask_first] * frac_pos[mask_first]))

        # Last boundary (cell_idx == self.ICP_window_size - 1)
        if np.any(mask_last):  # Check if there are any True values in mask_last
            np.add.at(Flux_perp_temp, self.ICP_window_size - 1, np.sum(ux[mask_last] * w[mask_last] * (1 - frac_pos[mask_last])))
            np.add.at(Flux_perp_temp, self.ICP_window_size, np.sum(ux[mask_last] * w[mask_last] * frac_pos[mask_last] * 2))

        # Middle cells (0 < cell_idx < self.ICP_window_size - 1)
        np.add.at(Flux_perp_temp, cell_idx[mask_middle], ux[mask_middle] * w[mask_middle] * (1 - frac_pos[mask_middle]))
        np.add.at(Flux_perp_temp, cell_idx[mask_middle] + 1, ux[mask_middle] * w[mask_middle] * frac_pos[mask_middle])

        # Send the flux to all processes
        Flux_perp = np.zeros_like(Flux_perp_temp)
        comm.Allreduce(Flux_perp_temp, Flux_perp, op=mpi.SUM)

        # Scale the flux to [1/(s*m^2)]
        Flux_perp /= self.dz

        return Flux_perp

    ###########################################################################
    # The next two functions have been deprecated and are not used in use     #
    ###########################################################################
    def calculate_J_perp_cells(self):
        '''
        Calculate the perpendicular conduction current density.
        '''
        # Get the current density of all species
        J_perp = np.zeros(self.ICP_window_size)
        for species in self.species_names:
            J_perp += self.charge_by_name[species] * self.calculate_Flux_perp_cells(species)

        return J_perp
    
    def calculate_Flux_perp_cells(self, species) -> np.ndarray[float]:
        '''
        Calculate the perpendicular flux density for a species.

        Parameters
        ----------
        species: str
            Name of species
        '''
        # Get wrappers for species
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle data
        try:
            ux = np.concatenate(species_wrapper.get_particle_ux())
            z = np.concatenate(species_wrapper.get_particle_z())
            w = np.concatenate(species_wrapper.get_particle_weight())
        except ValueError:
            ux = np.array([])
            z = np.array([])
            w = np.array([])

        # Calculate the flux of each cell in the ICP region
        Flux_perp_temp = np.zeros(self.ICP_window_size)

        idx = np.floor_divide(z, self.dz).astype(int) - self.zmin_idx
        mask = (idx >= 0) & (idx < self.ICP_window_size) # mask for particles in the ICP region
        valid_idx = idx[mask]
        weights = ux[mask] * w[mask]
        Flux_perp_temp[:len(np.bincount(valid_idx, weights))] += np.bincount(valid_idx, weights) # Use np.bincount to accumulate contributions

        # Send the flux to all processes
        Flux_perp = np.zeros_like(Flux_perp_temp)
        comm.Allreduce(Flux_perp_temp, Flux_perp, op=mpi.SUM)

        # Scale the flux to [1/(s*m^2)]
        Flux_perp /= self.dz

        return Flux_perp

    ###########################################################################
    # Support Functions                                                       #
    ###########################################################################
    def _save_data(self):
        '''
        Save the ICP diagnostic data to file.
        '''
        # Save the current density and field to file
        np.save(os.path.join(self.ICP_dir, f'Jperp_{self.curr_diag_output:04d}.npy'), self.Jperp_diag)
        np.save(os.path.join(self.ICP_dir, f'E_ICP_{self.curr_diag_output:04d}.npy'), self.field_diag)

class Diagnostics1D:
    def __init__(self,
                 simulation_obj: CapacitiveDischargeExample,
                 sim_ext: picmi.Simulation.extension,
                 switches: dict = None,
                 interval_times: list = None,
                 ion_spec_names: list = None,
                 diag_outfolder: str = './diags',
                 restart_checkpoint: bool = False
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
        switches: dict, optional
            Dictionary of switches for diagnostics
        interval_times: list, optional
            List of times to perform interval diagnostics, values must fall
            within the range [0, 1)
        ion_spec_names: list, optional
            List of ion species names
        diag_outfolder: str, optional
            Folder to save diagnostics
        restart_checkpoint: bool, optional
            Whether simulation is restarting from a checkpoint
        
        Notes
        -----
        - Interval times (if turned on in switches) need to be formatted like:

        interval_times = [time1, time2, ...]

        - Input switches need to be formatted like (default switches are shown):
        
        switches = {
            'ieadfs': {
                'z_lo': True,
                'z_hi': True,
            },
            'rate_ioniz' : False,
            'time_averaged': {
                'N_i': False,
                'N_e': False,
                'E_z': False,
                'phi': False,
                'W_e': False,
                'W_i': False,
                'Jze': False,
                'Jzi': False,
                'J_d': False,
                'CPe': False,
                'CPi': False,
                'IPe': False,
                'IPi': False,
                'EEdf': False,
                'IEdf': False
            },
            'time_resolved': {
                'N_i': True,
                'N_e': True,
                'E_z': False,
                'phi': True,
                'W_e': True,
                'W_i': True,
                'Jze': True,
                'Jzi': True,
                'J_d': True,
                'CPe': False,
                'CPi': False,
                'IPe': False,
                'IPi': False,
                'EEdf': False,
                'IEdf': False
            },
            'interval': {
                'N_i': False,
                'N_e': False,
                'E_z': False,
                'phi': False,
                'W_e': False,
                'W_i': False,
                'Jze': False,
                'Jzi': False,
                'J_d': False,
                'CPe': False,
                'CPi': False,
                'IPe': False,
                'IPi': False,
                'EEdf': False,
                'IEdf': False
            },
            'time_resolved_power': {
                'Pin_vst': False,
                'CPe_vst': False,
                'CPi_vst': False,
                'IPe_vst': False,
                'IPi_vst': False
            }
        }
        '''
        # Set up diagnostic switches
        if switches is None:
            ieadfs = {
                'z_lo': True,
                'z_hi': True,
            }
            self.Riz_switch = False
            time_averaged_dict = {
                'N_i': False,
                'N_e': False,
                'E_z': False,
                'phi': False,
                'W_e': False,
                'W_i': False,
                'Jze': False,
                'Jzi': False,
                'J_d': False,
                'CPe': False,
                'CPi': False,
                'IPe': False,
                'IPi': False,
                'EEdf': False,
                'IEdf': False
            }
            time_resolved_dict = {
                'N_i': True,
                'N_e': True,
                'E_z': False,
                'phi': True,
                'W_e': True,
                'W_i': True,
                'Jze': True,
                'Jzi': True,
                'J_d': True,
                'CPe': False,
                'CPi': False,
                'IPe': False,
                'IPi': False,
                'EEdf': False,
                'IEdf': False
            }
            interval_dict = {
                'N_i': False,
                'N_e': False,
                'E_z': False,
                'phi': False,
                'W_e': False,
                'W_i': False,
                'Jze': False,
                'Jzi': False,
                'J_d': False,
                'CPe': False,
                'CPi': False,
                'IPe': False,
                'IPi': False,
                'EEdf': False,
                'IEdf': False
            }
            self.tr_power_dict = {
                'Pin_vst': False,
                'CPe_vst': False,
                'CPi_vst': False,
                'IPe_vst': False,
                'IPi_vst': False
            }
        else:
            ieadfs = switches['ieadfs']
            self.Riz_switch = switches['rate_ioniz']
            time_averaged_dict = switches['time_averaged']
            time_resolved_dict = switches['time_resolved']
            interval_dict = switches['interval']
            self.tr_power_dict = switches['time_resolved_power']

        # Import simulation parameters
        self.m_ion = simulation_obj.m_ion
        self.rf_period = 1 / simulation_obj.freq
        self.in_period = simulation_obj.interval_time
        self.dt = simulation_obj.dt
        self.nz = simulation_obj.nz
        self.dz = simulation_obj.dz
        self.nodes = np.linspace(0, simulation_obj.gap, self.nz + 1)

        self.species_names = ['electrons']
        if ion_spec_names is not None:
            self.species_names += ion_spec_names

        # Set simulation extension object
        self.sim_ext = sim_ext

        # General diagnostics are collected in three tyes:
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

        self.num_outputs = simulation_obj.num_diag_steps
        self.diag_folder = diag_outfolder

        # Correct any power dictionary values
        if self.tr_power_dict['CPe_vst'] or self.tr_power_dict['Pin_vst']:
            time_resolved_dict['Jze'] = True
            time_resolved_dict['phi'] = True
        if self.tr_power_dict['CPi_vst'] or self.tr_power_dict['Pin_vst']:
            time_resolved_dict['Jzi'] = True
            time_resolved_dict['phi'] = True
        if self.tr_power_dict['IPe_vst']:
            time_resolved_dict['IPe'] = True
        if self.tr_power_dict['IPi_vst']:
            time_resolved_dict['IPi'] = True

        # Assemble master diagnostic dictionary
        self.master_diagnostic_dict = {
            'ieadfs': ieadfs,
            'time_averaged': time_averaged_dict,
            'time_resolved': time_resolved_dict,
            'interval': interval_dict
        }

        # Set dictionaries of charge and mass for particles
        self.mass_by_name = {
            'electrons': constants.m_e,
            ion_spec_names[0]: simulation_obj.m_ion
        }
        self.charge_by_name = {
            'electrons': -constants.q_e,
            ion_spec_names[0]: constants.q_e
        }
        
        self._make_particle_dictionaries()

        # Set up diagnostics
        self._import_general_timing_info(simulation_obj)
        self._get_time_resolved_steps(simulation_obj)
        if any(interval_dict.values()):
            self._get_interval_collection_steps()
        if self.Riz_switch:
            self._setup_Riz_diag(simulation_obj)
        self._calculate_N_collections()
        self._setup_diagnostic_arrays(simulation_obj)

        # Save settings to file
        self._save_diagnostic_inputs()
        self._save_edf_settings()
        self._save_cells_and_nodes(simulation_obj)

        # Set diagnostic output indices
        self.curr_diag_output = 0
        self.curr_tr = 0
        self.curr_ta = 0
        self.curr_interval = 0
        self.curr_slice = 0

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
        self.ieadf_by_species = {}
        for species in self.species_names[1:]:
            self.ieadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['ieadfs'].items():
                if value:
                    self.ieadf_by_species[species][key] = np.zeros((len(self.ieadf_bin_centers), len(self.iadf_bin_centers)))

        # Ionization rate arrays
        if self.Riz_switch:
            self.Riz_by_species = {}
            for species in self.species_names[1:]:
                self.Riz_by_species[species] = np.zeros((self.Riz_nt, self.nz))

        # Create eedf and iedf bins
        self.eedf_bin_edges = np.linspace(0, simulation_obj.eedf_max_eV, simulation_obj.num_bins + 1)
        self.eedf_bin_centers = np.multiply(self.eedf_bin_edges[:-1] + self.eedf_bin_edges[1:], 0.5)
        self.iedf_bin_edges = np.linspace(0, simulation_obj.iedf_max_eV, simulation_obj.num_bins + 1)
        self.iedf_bin_centers = np.multiply(self.iedf_bin_edges[:-1] + self.iedf_bin_edges[1:], 0.5)

        # Time resolved arrays
        self.tr_N_e = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_N_i = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_W_e = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_W_i = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_E_z = np.zeros((self.tr_coll[0], self.nz))
        self.tr_phi = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_Jze = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_Jzi = np.zeros((self.tr_coll[0], self.nz + 1))
        self.tr_J_d = np.zeros((self.tr_coll[0], self.nz))
        self.tr_CPe = np.zeros((self.tr_coll[0], self.nz))
        self.tr_CPi = np.zeros((self.tr_coll[0], self.nz))
        self.tr_IPe = np.zeros((self.tr_coll[0], self.nz))
        self.tr_IPi = np.zeros((self.tr_coll[0], self.nz))
        self.tr_EEdf = np.zeros((self.tr_coll[0], len(self.eedf_bin_centers)))
        self.tr_IEdf = np.zeros((self.tr_coll[0], len(self.iedf_bin_centers)))
        self.tr_times = np.zeros((self.tr_coll[0]))

        # Power arrays
        self.tr_Pin_vst = None
        self.tr_CPe_vst = None
        self.tr_CPi_vst = None
        self.tr_IPe_vst = None
        self.tr_IPi_vst = None

        # Time averaged arrays
        self.ta_N_e = np.zeros(self.nz + 1)
        self.ta_N_i = np.zeros(self.nz + 1)
        self.ta_W_e = np.zeros(self.nz + 1)
        self.ta_W_i = np.zeros(self.nz + 1)
        self.ta_E_z = np.zeros(self.nz)
        self.ta_phi = np.zeros(self.nz + 1)
        self.ta_Jze = np.zeros(self.nz + 1)
        self.ta_Jzi = np.zeros(self.nz + 1)
        self.ta_J_d = np.zeros(self.nz)
        self.ta_CPe = np.zeros(self.nz)
        self.ta_CPi = np.zeros(self.nz)
        self.ta_IPe = np.zeros(self.nz)
        self.ta_IPi = np.zeros(self.nz)
        self.ta_EEdf = np.zeros(len(self.eedf_bin_centers))
        self.ta_IEdf = np.zeros(len(self.iedf_bin_centers))

        # Interval arrays
        self.in_N_e = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_N_i = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_W_e = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_W_i = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_E_z = np.zeros((len(self.in_slices), self.nz))
        self.in_phi = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_Jze = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_Jzi = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_J_d = np.zeros((len(self.in_slices), self.nz))
        self.in_CPe = np.zeros((len(self.in_slices), self.nz))
        self.in_CPi = np.zeros((len(self.in_slices), self.nz))
        self.in_IPe = np.zeros((len(self.in_slices), self.nz))
        self.in_IPi = np.zeros((len(self.in_slices), self.nz))
        self.in_EEdf = np.zeros((len(self.in_slices), len(self.eedf_bin_centers)))
        self.in_IEdf = np.zeros((len(self.in_slices), len(self.iedf_bin_centers)))

        # Single diagnostic output arrays
        # Array of diagnostic species indices
        self.diag_idx_by_name = {}
        for i in range(len(self.species_names)):
            self.diag_idx_by_name[self.species_names[i]]=i

        # Density & Temperature - first column is electron density, rest is for ions
        self.N = []
        for i in range(len(self.species_names)):
            self.N.append(np.zeros(self.nz + 1))
        self.N = np.stack(self.N)

        self.W = []
        for i in range(len(self.species_names)):
            self.W.append(np.zeros(self.nz + 1))
        self.W = np.stack(self.W)

        self.J = []
        for i in range(len(self.species_names)):
            self.J.append(np.zeros(self.nz + 1))
        self.J = np.stack(self.J)

        self.J_d = []
        for i in range(len(self.species_names)):
            self.J_d.append(np.zeros(self.nz))
        self.J_d = np.stack(self.J_d)

        self.P_C = []
        for i in range(len(self.species_names)):
            self.P_C.append(np.zeros(self.nz))
        self.P_C = np.stack(self.P_C)

        self.P_I = []
        for i in range(len(self.species_names)):
            self.P_I.append(np.zeros(self.nz))
        self.P_I = np.stack(self.P_I)

        self.Edf = []
        for i in range(len(self.species_names)):
            if i == 0:
                self.Edf.append(np.zeros(len(self.eedf_bin_centers)))
            else:
                self.Edf.append(np.zeros(len(self.iedf_bin_centers)))
        self.Edf = np.stack(self.Edf)

        self.E = np.zeros(self.nz)
        self.phi = np.zeros(self.nz + 1)
        self.E_last_step = np.zeros(self.nz)
    
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
            self.ta_coll[ii] = int(total_steps + 1)
        
        if comm.rank != 0:
            return
        
        # Save the number of collections to file
        self.check_file(f'{self.diag_folder}/N_collections.dat')
        with open(f'{self.diag_folder}/N_collections.dat', 'w') as f:
            f.write('Number of Collections\n')
            f.write('---------------------\n')
            f.write('Diagnostic Output, Time Resolved, Time Averaged\n')
            for ii in range(self.num_outputs):
                f.write(f'{ii}, {self.tr_coll[ii]}, {self.ta_coll[ii]}\n')

    def _setup_Riz_diag(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Set up diagnostics for ionization rate
        
        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        # Import simulation parameters
        self.Riz_collection_time = simulation_obj.Riz_collection_time

        # Set up ionization rate collection time
        if self.Riz_collection_time <= 0:
            raise ValueError('Riz_collection_time must be greater than zero.')

        self.diag_cycle_time = self.diag_time + self.evolve_time

        # Calculate number of diagnostic collections per Riz collection
        self.diag_collections_per_Riz = self.Riz_collection_time // self.diag_cycle_time

        if self.diag_collections_per_Riz == 0:
            self.diag_collections_per_Riz = 1
            self.Riz_collection_time = self.diag_cycle_time
        elif self.diag_collections_per_Riz > self.num_outputs:
            self.diag_collections_per_Riz = self.num_outputs
            self.Riz_collection_time = self.diag_collections_per_Riz * self.diag_cycle_time
        else:
            self.Riz_collection_time = self.diag_collections_per_Riz * self.diag_cycle_time

        # Initialize variables/counters for ionization rate collection
        self.Riz_diag_counter = 0
        self.Riz_max_output = self.num_outputs // self.diag_collections_per_Riz
        self.Riz_current_output = 1
        self.Riz_start_time = self.diag_start_time

        # Calculate the unit size for time and space discretization
        self.Riz_dz = self.dz

        # Choose the time step to be either dt, or rf_period/nz
        if int(self.rf_period / self.dt) < self.nz:
            self.Riz_nt = int(self.rf_period / self.dt)
        else:
            self.Riz_nt = self.nz

        self.R_inoiz_dt = self.rf_period / self.Riz_nt

        # Set up the ionization rate directory
        if comm.rank != 0:
            return

        # Make a diagnostics directory
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)

        # Make an ionization rate directory for each ion species
        self.Riz_dir_by_species = {}
        for species in self.species_names[1:]:
            self.Riz_dir_by_species[species] = os.path.join(self.diag_folder, f'r_ioniz_{species}')
            if not os.path.exists(self.Riz_dir_by_species[species]):
                os.makedirs(self.Riz_dir_by_species[species])

        # Save the ionization rate grid
        Riz_z_edges = np.linspace(0, simulation_obj.gap, self.nz + 1)
        Riz_time_edges = np.linspace(0, 1, self.Riz_nt + 1)
        Riz_z_centers = np.multiply(Riz_z_edges[:-1] + Riz_z_edges[1:], 0.5)
        Riz_time_centers = np.multiply(Riz_time_edges[:-1] + Riz_time_edges[1:], 0.5)

        for species in self.species_names[1:]:
            # Check if file exists
            self.check_file(f'{self.Riz_dir_by_species[species]}/bins_t.npy')
            self.check_file(f'{self.Riz_dir_by_species[species]}/bins_z.npy')
            np.save(f'{self.Riz_dir_by_species[species]}/bins_z.npy', Riz_z_centers)
            np.save(f'{self.Riz_dir_by_species[species]}/bins_t.npy', Riz_time_centers)

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
        Set up an array containing steps to calculate interval diagnostics.
        
        Sets up a list with length equal to the number of diagnostic
        outputs. Each element of the list is a stack of numpy arrays
        containing the steps at which interval diagnostics are to be
        performed for that diagnostic output. The length of the stack
        is equal to the number of full intervals that fit within the
        diagnostic output window.

        Example
        -------
        Suppose we have 3 diagnostic outputs and can fit 4 intervals
        within each diagnostic window. Then the list will obey:
        
        ```
        len(self.in_coll_steps) = 3
        ```

        and for `ii` in `[0, 1, 2]`:

        ```
        len(self.in_coll_steps[ii]) = 4
        ```
        '''
        self.in_coll_steps = []

        for ii in range(self.num_outputs):
            # Start time of current diag output window
            output_start_t = self.diag_start[ii] * self.dt
            output_end_t = self.diag_stop[ii] * self.dt

            # Initialize collection times
            collection_times = []

            # Count number of periods before the diagnostic output
            period_start_collection = int(output_start_t // self.in_period)

            # Get collection times
            # NOTE: We only collect for intervals that are fully within the diagnostic output
            temp_collec_times = (period_start_collection + self.in_slices) * self.in_period
            while any(temp_collec_times < output_start_t):
                temp_collec_times += self.in_period
                if any(temp_collec_times > output_end_t):
                    self.in_coll_steps.append(np.array([]))
                    break

            # Append the first valid collection time
            collection_times.append(temp_collec_times.copy())

            # Get the times of each interval until output_end_t
            while collection_times[-1][-1] < output_end_t:
                temp_collec_times += self.in_period
                collection_times.append(temp_collec_times.copy())

            # Remove the last collection time if it is beyond the output_end_t
            if collection_times[-1][-1] > output_end_t:
                collection_times.pop()

            # Convert the list of times to a numpy stack
            if collection_times:
                collection_times = np.stack(collection_times)
            else:
                collection_times = np.array([])

            # Convert times to steps
            collection_steps = np.round(collection_times / self.dt).astype(int)

            # Add the steps to the collection array
            self.in_coll_steps.append(collection_steps)

        # Save the interval collection steps to file
        if comm.rank != 0:
            return

        # Check if the folder exists
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)
        file = os.path.join(self.diag_folder, 'intrvl_collection_steps.dat')
        with open(file, 'w') as f:
            f.write('Interval Collection Steps\n')
            for ii in range(self.num_outputs):
                f.write(f'\nDiagnostic Output #{ii+1}\n')
                f.write(f'---------------------\n')
                for jj in range(len(self.in_coll_steps[ii])):
                    f.write(f'Interval #{jj+1}:\n')
                    f.write(f'    {self.in_coll_steps[ii][jj]}\n')

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
            f.write(f'Cell size [m]={self.dz}\n\n')

            f.write('Diagnostic Parameters\n')
            f.write('---------------------\n')
            f.write(f'Diagnostics start time [s]={self.diag_start_time}\n')
            f.write(f'Diagnostic time [s]={self.diag_time}\n')
            f.write(f'Evolve time [s]={self.evolve_time}\n\n')
            
            f.write(f'Number of diagnostic outputs={self.num_outputs}\n\n')

            f.write(f'Time [s] between time resolved collections={self.tr_interval}\n')
            f.write(f'Time resolved collections per diagnostic={self.num_in_tr}\n\n')

            f.write(f'Interval period [s]={self.in_period}\n')
            f.write(f'Times in interval={', '.join(map(str,self.in_slices))}\n\n')

            f.write(f'Output #   |   Start Step   |   Stop Step   |   Start Time   |   Stop Time\n')
            f.write(f'------------------------------------------------------------------------------\n')
            for ii in range(self.num_outputs):
                f.write(f'   {ii+1:5d}   | {self.diag_start[ii]:12d}   |{self.diag_stop[ii]:12d}   | {self.diag_start[ii]*self.dt:.8e} | {self.diag_stop[ii]*self.dt:.8e}\n')

    def _save_edf_settings(self):
        '''
        Save the settings for energy distribution function creation
        '''
        if comm.rank != 0:
            return

        if any(self.master_diagnostic_dict['ieadfs'].values()) or any(self.master_diagnostic_dict[key].get(metric) for key in self.master_diagnostic_dict for metric in ['EEdf', 'IEdf']):
            # Make a diagnostics directory
            if not os.path.exists(self.diag_folder):
                os.makedirs(self.diag_folder)

        # Save the wall IEADF settings
        if any(self.master_diagnostic_dict['ieadfs'].values()):
            # Make an ieadf directory for each ion species
            self.ieadf_dir_by_species = {}
            for species in self.species_names[1:]:
                self.ieadf_dir_by_species[species] = os.path.join(self.diag_folder, f'ieadf_{species}')
                if not os.path.exists(self.ieadf_dir_by_species[species]):
                    os.makedirs(self.ieadf_dir_by_species[species])

            # Save the ieadf energy bins
            for species in self.species_names[1:]:
                # Check if file exists
                self.check_file(f'{self.ieadf_dir_by_species[species]}/bins_eV.npy')
                self.check_file(f'{self.ieadf_dir_by_species[species]}/bins_deg.npy')
                np.save(f'{self.ieadf_dir_by_species[species]}/bins_eV.npy', self.ieadf_bin_centers)
                np.save(f'{self.ieadf_dir_by_species[species]}/bins_deg.npy', self.iadf_bin_centers)

        # Save the normal EDF settings
        if any(dict.get('EEdf') for dict in self.master_diagnostic_dict.values()):
            # Save the eedf energy bins
            self.check_file(f'{self.diag_folder}/eedf_bins_eV.npy')
            np.save(f'{self.diag_folder}/eedf_bins_eV.npy', self.eedf_bin_centers)

        if any(dict.get('IEdf') for dict in self.master_diagnostic_dict.values()):
            self.check_file(f'{self.diag_folder}/iedf_bins_eV.npy')
            np.save(f'{self.diag_folder}/iedf_bins_eV.npy', self.iedf_bin_centers)

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
        self.check_file('diags/nodes.npy')
        np.save('diags/nodes.npy', self.nodes)

        # Make a npy file of cell centers
        z = np.linspace(self.dz / 2, simulation_obj.gap - self.dz / 2, self.nz)
        # Check if file exists
        self.check_file('diags/cells.npy')
        np.save('diags/cells.npy', z)

    def _make_particle_dictionaries(self):
        '''
        Make dictionaries with keys self.species_names for diag indices,
        mass, and charge.
        '''
        # Make array of species diagnostic indices
        self.diag_idx_by_name = {}
        for i in range(len(self.species_names)):
            self.diag_idx_by_name[self.species_names[i]]=i
    
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

    #     # Get index of species array in stack
    #     idx = self.diag_idx_by_name[species]

    #     # Report the density
    #     rho_data = rho_wrapper[...]
    #     self.N[idx] = rho_data

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

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        # Report the current
        self.N[idx] = N_data

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

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        # Report the temperature
        self.W[idx] = W_data

    def update_Jz(self, species):
        '''
        Return current density [A/m^2] at cells (this is so that we can
        cleanly do J*E, since E is calculated within cells) for a species.
        Needs to be multiplied by charge and divided by cell size before
        being used.

        Parameters
        ----------
        species: str
            Name of species
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

        # Get cell index of particle
        cell_idx = np.floor(z / self.dz).astype(int)

        # Calculate the fractional position within the cell
        frac_pos = (z / self.dz) - cell_idx

        # Calculate weights for interpolation
        frac_l = 1 - frac_pos
        frac_r = frac_pos

        # Sort by z and assign uz to nodes
        temp_J = np.zeros(self.nz + 1)
        np.add.at(temp_J, cell_idx, uz * w * frac_l)
        # Get a list of all particles which are not at the last node
        valid_idxs = cell_idx != self.nz
        np.add.at(temp_J, cell_idx[valid_idxs] + 1, uz[valid_idxs] * w[valid_idxs] * frac_r[valid_idxs])

        # Future note: If we want to get rid of the dip in data reported at the first and last nodes,
        #              we can create a mask of particles in the first and last cell and double their
        #              contributions to the first and last node.

        # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_J to all processes
        J_data = np.zeros_like(temp_J)
        comm.Allreduce(temp_J, J_data, op=mpi.SUM)

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        # Report the current
        self.J[idx] = J_data

    def update_E(self):
        '''
        Return electric field at node points
        '''
        E_wrapper = fields.EzFPWrapper()
        self.E = E_wrapper[...]

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
        '''
        # Set up wrappers
        species_wrapper = particle_containers.ParticleContainerWrapper(species)

        # Get particle velocities
        try:
            ux = np.concatenate(species_wrapper.get_particle_ux())
            w = np.concatenate(species_wrapper.get_particle_weight())
            z = np.concatenate(species_wrapper.get_particle_z())
        except ValueError:
            ux = np.array([])
            w = np.array([])
            z = np.array([])

        # Get the perpendicular field 
        Ex_nodes = fields.ExFPWrapper()

        # Field is on the nodes, so average it out to the cell centers
        Ex_centers = (Ex_nodes[:-1] + Ex_nodes[1:]) / 2

        # Get cell index of particles
        cell_idx = np.floor(z / self.dz).astype(int)

        # Calculate the fractional position within the cell
        frac_pos = (z / self.dz) - cell_idx

        # Initialize the array of the field at each particle position
        Ex_at_particle = np.zeros(len(z))

        # Create masks to classify the particles
        mask_low_edge = (cell_idx == 0) & (frac_pos <= 0.5)
        mask_high_edge = (cell_idx == self.nz - 1) & (frac_pos >= 0.5) | (cell_idx == self.nz)
        # Ensure that the edge cases are not picked up by the other masks
        mask_before_center = (frac_pos < 0.5) & ~(mask_low_edge | mask_high_edge)
        mask_after_center = (frac_pos >= 0.5) & ~(mask_low_edge | mask_high_edge)

        # Handle particles near the low edge (index 0)
        Ex_at_particle[mask_low_edge] = Ex_centers[0]

        # Handle particles near the high edge (index nz - 1) and
        # revert all cell indices at the high edge to the last cell (this
        # prevents out of bounds errors for particles exactly at the boundary)
        cell_idx[mask_high_edge] = self.nz - 1
        Ex_at_particle[mask_high_edge] = Ex_centers[self.nz - 1]

        # Handle particles before the center of the cell
        rel_position_before = frac_pos[mask_before_center] + 0.5
        Ex_at_particle[mask_before_center] = (
            Ex_centers[cell_idx[mask_before_center] - 1] +
            (Ex_centers[cell_idx[mask_before_center]] - Ex_centers[cell_idx[mask_before_center] - 1]) * rel_position_before
        )

        # Handle particles after the center of the cell
        rel_position_after = frac_pos[mask_after_center] - 0.5
        Ex_at_particle[mask_after_center] = (
            Ex_centers[cell_idx[mask_after_center]] +
            (Ex_centers[cell_idx[mask_after_center] + 1] - Ex_centers[cell_idx[mask_after_center]]) * rel_position_after
        )

        # # Commenting this out, but writing out how to do a linear interpolation
        # # for the external particles, incase I find out this is what WarpX does
        # first_position = 1.5 - frac_pos
        # Ex_at_particle[first_parts] = Ex_centers[1] - (Ex_centers[1] - Ex_centers[0]) * first_position
        # end_position = 0.5 + frac_pos
        # Ex_at_particle[end_parts] = Ex_centers[self.nz - 2] + (Ex_centers[self.nz - 1] - Ex_centers[self.nz - 2]) * end_position

        # Sort by z and assign power input to cells
        temp_P = np.zeros(self.nz)
        np.add.at(temp_P, cell_idx, ux * Ex_at_particle * w)

        # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_P to all processes
        P_data = np.zeros_like(temp_P)
        comm.Allreduce(temp_P, P_data, op=mpi.SUM)

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        # Report the temperature
        self.P_I[idx] = P_data

    def update_P_C(self, species):
        '''
        Calculate power into plasma via capacitive heating. Needs to be
        multiplied by charge and divided by cell size before being used.

        This interpolates the field to the particle positions using a linear
        shape, similar to what WarpX does. Minor differences in the two methods
        (e.g. I don't know exactly how WarpX interpolates for particles at the
        boundary) may lead to slight differences from the actual power.
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
        Ez_centers = fields.EzFPWrapper()
        Ez_centers = Ez_centers[...]

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

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        # Report the temperature
        self.P_C[idx] = P_data

    def update_J_d(self):
        '''
        Calculate the displacement current density. Needs be multiplied by
        constants.ep0 and divided by the time step before being used.

        We use a backward difference to calculate the displacement current
        and CANNOT use this implementation at the first time step.
        '''
        # Save the electric field from the current time step, if not already done
        if not any(dict.get('E_z') for dict in self.master_diagnostic_dict.values()):
            E_wrapper = fields.EzFPWrapper()
            self.E = E_wrapper[...]

        # Calculate the displacement current density
        self.J_d = self.E - self.E_last_step

    def calculate_eedf(self):
        '''
        Gets a histogram of the electron energy distribution function.
    
        Returns
        -------
        hist: np.ndarray
            The histogram of the electron energy distribution function
        '''
        def get_eedf():
            '''
            Gets the electron energy distribution function.
            '''
            # Set up wrappers
            species_wrapper = particle_containers.ParticleContainerWrapper('electrons')

            try:
                ux = np.concatenate(species_wrapper.get_particle_ux())
                uy = np.concatenate(species_wrapper.get_particle_uy())
                uz = np.concatenate(species_wrapper.get_particle_uz())
                w  = np.concatenate(species_wrapper.get_particle_weight())
            except ValueError:
                ux = np.array([])
                uy = np.array([])
                uz = np.array([])
                w = np.array([])

            # Calculate the energy
            v2 = (np.square(ux) + np.square(uy) + np.square(uz))
            E = np.multiply(v2, 0.5 * constants.m_e / constants.q_e)

            # Get the histogram (unnormalized)
            hist, *_ = np.histogram(E, bins=self.eedf_bin_edges, density=False, weights=w/self.dz)

            hist = np.copy(hist, order='C')

            return hist
    
        # Get the ieadf on the processor
        hist = get_eedf()

        # Sum the ieadf histograms from all processors
        hist_all = np.zeros_like(hist)
        comm.Allreduce(hist, hist_all, op=mpi.SUM)

        # Get index of species array in stack
        idx = self.diag_idx_by_name['electrons']

        self.Edf[idx] = hist_all

    def calculate_iedf(self, species: str):
        '''
        Gets a histogram of the ion energy distribution function.
        
        Parameters
        ----------
        species: str
            The name of the ion species for which to calculate the
            distribution function

        Returns
        -------
        hist: np.ndarray
            The histogram of the ion energy distribution function
        '''
        def get_iedf(species):
            '''
            Gets the ion energy distribution function.
            '''
            # Set up wrappers
            species_wrapper = particle_containers.ParticleContainerWrapper(species)

            try:
                ux = np.concatenate(species_wrapper.get_particle_ux())
                uy = np.concatenate(species_wrapper.get_particle_uy())
                uz = np.concatenate(species_wrapper.get_particle_uz())
                w  = np.concatenate(species_wrapper.get_particle_weight())
            except ValueError:
                ux = np.array([])
                uy = np.array([])
                uz = np.array([])
                w = np.array([])

            # Calculate the energy
            v2 = (np.square(ux) + np.square(uy) + np.square(uz))
            E = np.multiply(v2, 0.5 * self.m_ion / constants.q_e)

            # Get the histogram (unnormalized)
            hist, *_ = np.histogram(E, bins=self.iedf_bin_edges, density=False, weights=w/self.dz)

            hist = np.copy(hist, order='C')

            return hist
    
        # Get the ieadf on the processor
        hist = get_iedf(species)

        # Sum the ieadf histograms from all processors
        hist_all = np.zeros_like(hist)
        comm.Allreduce(hist, hist_all, op=mpi.SUM)

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        self.Edf[idx] = hist_all

    def calculate_ieadf(self, species: str, boundary: str):
        '''
        Gets a histogram of the ion energy angular distribution function at
        the specified boundary for the energy bins self.iedf_bin_centers.
        
        Parameters
        ----------
        species: str
            The name of the species for which to calculate the ion energy
        boundary: str
            The boundary at which to calculate the ion energy distribution
            function, one of 'z_lo', 'z_hi'
        
        Returns
        -------
        hist: np.ndarray
            The histogram of the ion energy distribution function
        '''
        def get_ieadf(species, boundary):
            '''
            Gets the ion energy angular distribution function.
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
            E = np.multiply(v2, 0.5 * self.m_ion / constants.q_e)

            # Calculate the ion xy velocity
            vxy = np.sqrt(np.square(ux) + np.square(uy))
            # Calculate angle with a negative sign so that left/right wall ieadfs are on the left/right of an energy vs angle plot
            angle = np.arctan(vxy / uz) * 180 / np.pi

            # Get the histogram (unnormalized)
            hist, *_ = np.histogram2d(E, angle, bins=[self.ieadf_bin_edges, self.iadf_bin_edges], density=False, weights=w/self.dz)

            # hist = np.ascontiguousarray(hist, dtype=np.float64)
            hist = np.copy(hist, order='C')

            return hist
    
        # Get the ieadf on the processor
        hist = get_ieadf(species, boundary)

        # Sum the ieadf histograms from all processors
        hist_all = np.zeros_like(hist)
        comm.Allreduce(hist, hist_all, op=mpi.SUM)
        self.ieadf_by_species[species][boundary] = hist_all
    
    def clear_ieadf_buffers(self):
        '''
        Clears the buffers for the ion energy angular distribution function.
        '''
        # Clear the boundary buffers
        boundary_wrapper = particle_containers.ParticleBoundaryBufferWrapper()
        boundary_wrapper.clear_buffer()

    def add_buff_Riz(self):
        '''
        Adds the boundary buffers to the ionization rate. Use this if clearing
        the boundary buffers before calculating the ionization rate.
        '''
        for spec in self.species_names[1:]:
            # Set up wrappers
            bd_wrapper = particle_containers.ParticleBoundaryBufferWrapper()

            # Get boundary particle data
            bd_t = {}
            bd_z = {}
            bd_w = {}
            for boundary in ['z_lo', 'z_hi']:
                try:
                    bd_t[boundary] = np.concatenate(bd_wrapper.get_particle_boundary_buffer(spec, boundary, 'orig_t', 0))
                    bd_z[boundary] = np.concatenate(bd_wrapper.get_particle_boundary_buffer(spec, boundary, 'orig_z', 0))
                    bd_w[boundary] = np.concatenate(bd_wrapper.get_particle_boundary_buffer(spec, boundary, 'w', 0))
                except ValueError:
                    bd_t[boundary] = np.array([])
                    bd_z[boundary] = np.array([])
                    bd_w[boundary] = np.array([])        

            # Make the current time the end of the time window
            t_begin = self.Riz_start_time + self.diag_cycle_time * self.Riz_diag_counter

            # Now, make a histogram of the ionization rate
            for z, t, w in zip(np.concatenate([bd_z['z_lo'], bd_z['z_hi']]), np.concatenate([bd_t['z_lo'], bd_t['z_hi']]), np.concatenate([bd_w['z_lo'], bd_w['z_hi']])):
                # Check if the particle is in the time window
                if t < t_begin:
                    continue

                # Get the cell index
                z_idx = int(z / self.dz)

                # Get the time index
                t_idx = int(t / self.dt) % self.Riz_nt

                # Increment the ionization rate
                try:
                    self.Riz_by_species[spec][t_idx][z_idx] += w
                except IndexError:
                    self.Riz_by_species[spec][t_idx][z_idx - 1] += w

    def add_bulk_Riz(self):
        '''
        Calculate the ionization rate for each bulk species.
        '''
        for spec in self.species_names[1:]:
            # Set up wrappers
            sp_wrapper = particle_containers.ParticleContainerWrapper(spec)

            # Get bulk particle data
            try:
                bk_t = np.concatenate(sp_wrapper.get_particle_real_arrays('orig_t', 0))
                bk_z = np.concatenate(sp_wrapper.get_particle_real_arrays('orig_z', 0))
                bk_w = np.concatenate(sp_wrapper.get_particle_weight())
            except ValueError:
                bk_t = np.array([])
                bk_z = np.array([])
                bk_w = np.array([])

            t_begin = self.Riz_start_time + self.diag_cycle_time * self.Riz_diag_counter

            # Now, make a histogram of the ionization rate
            for z, t, w in zip(bk_z, bk_t, bk_w):
                # Check if the particle is in the time window
                if t < t_begin:
                    continue

                # Get the cell index
                z_idx = int(z / self.dz)

                # Get the time index
                t_idx = int(t / self.dt) % self.Riz_nt

                # Increment the ionization rate
                self.Riz_by_species[spec][t_idx][z_idx] += w

    def collect_Riz(self):
        '''
        Collect the ionization rate from all processors.
        '''
        for spec in self.species_names[1:]:
            # Sum the ionization rate histograms from all processors
            Riz = np.zeros_like(self.Riz_by_species[spec])
            comm.Allreduce(self.Riz_by_species[spec], Riz, op=mpi.SUM)
            self.Riz_by_species[spec] = Riz

    ###########################################################################
    # Simulation Functions                                                    #
    ###########################################################################
    def do_diagnostics(self):
        '''
        Master function to perform diagnostics at each time step. Should be
        installed at least one step before the first diagnostic step.
        '''        
        def do_time_resolved_diagnostics(tr_idx: int):
            '''
            Performs time resolved diagnostics

            Parameters
            ----------
            tr_idx: int
                Index of the time resolved diagnostic
            '''
            # Grab temporary dictionary for time resolved diagnostics
            temp_settings = self.master_diagnostic_dict['time_resolved']

            # Add values
            if temp_settings['N_e']:
                self.tr_N_e[tr_idx] = self.N[0]
            if temp_settings['N_i']:
                self.tr_N_i[tr_idx] = self.N[1]
            if temp_settings['W_e']:
                self.tr_W_e[tr_idx] = self.W[0]
            if temp_settings['W_i']:
                self.tr_W_i[tr_idx] = self.W[1]
            if temp_settings['E_z']:
                self.tr_E_z[tr_idx] = self.E
            if temp_settings['phi']:
                self.tr_phi[tr_idx] = self.phi
            if temp_settings['Jze']:
                self.tr_Jze[tr_idx] = self.J[0]
            if temp_settings['Jzi']:
                self.tr_Jzi[tr_idx] = self.J[1]
            if temp_settings['J_d']:
                self.tr_J_d[tr_idx] = self.J_d
            if temp_settings['CPe']:
                self.tr_CPe[tr_idx] = self.P_C[0]
            if temp_settings['CPi']:
                self.tr_CPi[tr_idx] = self.P_C[1]
            if temp_settings['IPe']:
                self.tr_IPe[tr_idx] = self.P_I[0]
            if temp_settings['IPi']:
                self.tr_IPi[tr_idx] = self.P_I[1]
            if temp_settings['EEdf']:
                self.tr_EEdf[tr_idx] = self.Edf[0]
            if temp_settings['IEdf']:
                self.tr_IEdf[tr_idx] = self.Edf[1]

            # Add time to time array
            self.tr_times[tr_idx] = self.sim_ext.warpx.gett_new(lev=0)

        def do_time_averaged_diagnostics():
            '''
            Performs time averaged diagnostics
            '''
            # Grab temporary dictionary for time averaged diagnostics
            temp_settings = self.master_diagnostic_dict['time_averaged']

            # Add values now, average later
            if temp_settings['N_e']:
                self.ta_N_e += self.N[0]
            if temp_settings['N_i']:
                self.ta_N_i += self.N[1]
            if temp_settings['W_e']:
                self.ta_W_e += self.W[0]
            if temp_settings['W_i']:
                self.ta_W_i += self.W[1]
            if temp_settings['E_z']:
                self.ta_E_z += self.E
            if temp_settings['phi']:
                self.ta_phi += self.phi
            if temp_settings['Jze']:
                self.ta_Jze += self.J[0]
            if temp_settings['Jzi']:
                self.ta_Jzi += self.J[1]
            if temp_settings['J_d']:
                self.ta_J_d += self.J_d
            if temp_settings['CPe']:
                self.ta_CPe += self.P_C[0]
            if temp_settings['CPi']:
                self.ta_CPi += self.P_C[1]
            if temp_settings['IPe']:
                self.ta_IPe += self.P_I[0]
            if temp_settings['IPi']:
                self.ta_IPi += self.P_I[1]
            if temp_settings['EEdf']:
                self.ta_EEdf += self.Edf[0]
            if temp_settings['IEdf']:
                self.ta_IEdf += self.Edf[1]

        def do_interval_diagnostics(interval_idx: int):
            '''
            Perform diagnostics at an time within interval self.interval_time

            Parameters
            ----------
            interval_idx: int
                Index of interval in self.times_in_interval. Determines which array
                to update.
            '''
            # Grab temporary dictionary for time averaged diagnostics
            temp_settings = self.master_diagnostic_dict['interval']

            # Add values now, average later
            if temp_settings['N_e']:
                self.in_N_e[interval_idx] += self.N[0]
            if temp_settings['N_i']:
                self.in_N_i[interval_idx] += self.N[1]
            if temp_settings['W_e']:
                self.in_W_e[interval_idx] += self.W[0]
            if temp_settings['W_i']:
                self.in_W_i[interval_idx] += self.W[1]
            if temp_settings['E_z']:
                self.in_E_z[interval_idx] += self.E
            if temp_settings['phi']:
                self.in_phi[interval_idx] += self.phi
            if temp_settings['Jze']:
                self.in_Jze[interval_idx] += self.J[0]
            if temp_settings['Jzi']:
                self.in_Jzi[interval_idx] += self.J[1]
            if temp_settings['J_d']:
                self.in_J_d[interval_idx] += self.J_d
            if temp_settings['CPe']:
                self.in_CPe[interval_idx] += self.P_C[0]
            if temp_settings['CPi']:
                self.in_CPi[interval_idx] += self.P_C[1]
            if temp_settings['IPe']:
                self.in_IPe[interval_idx] += self.P_I[0]
            if temp_settings['IPi']:
                self.in_IPi[interval_idx] += self.P_I[1]
            if temp_settings['EEdf']:
                self.in_EEdf[interval_idx] += self.Edf[0]
            if temp_settings['IEdf']:
                self.in_IEdf[interval_idx] += self.Edf[1]

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
            # Clear the ieadf buffers for this collection
            self.clear_ieadf_buffers()

        # Synchronize, if necessary, to catch velocities up to positions
        if any(self.master_diagnostic_dict[key].get(metric) for key in self.master_diagnostic_dict for metric in ['Jze', 'Jzi', 'CPe', 'CPi', 'IPe', 'IPi', 'W_e', 'W_i']):
            self.sim_ext.warpx.synchronize()

        # Check if we need to save the electric field for the displacement current
        save_E_last_step = False
        if self.master_diagnostic_dict['time_resolved']['J_d'] and (next_step - self.diag_start[self.curr_diag_output]) % self.diag_time_resolving_steps == 0:
            save_E_last_step = True
        if self.master_diagnostic_dict['time_averaged']['J_d'] and (next_step >= self.diag_start[self.curr_diag_output]):
            save_E_last_step = True
        if len(self.in_coll_steps[self.curr_diag_output]) > 0:
            if self.master_diagnostic_dict['interval']['J_d'] and next_step == self.in_coll_steps[self.curr_diag_output][self.curr_interval][self.curr_slice]:
                save_E_last_step = True

        # Go through each diagnostic type and determine if we need to update
        # arrays for that diagnostic at this time step
        time_resolved = False
        time_averaged = False
        interval = False
        if any(self.master_diagnostic_dict['time_resolved'].values()) and ((step - self.diag_start[self.curr_diag_output]) % self.diag_time_resolving_steps == 0) and step >= self.diag_start[self.curr_diag_output]:
            time_resolved = True
        if any(self.master_diagnostic_dict['time_averaged'].values()) and (step >= self.diag_start[self.curr_diag_output]):
            time_averaged = True
        if any(self.master_diagnostic_dict['interval'].values()) and len(self.in_coll_steps[self.curr_diag_output]) > 0:
            if (step == self.in_coll_steps[self.curr_diag_output][self.curr_interval][self.curr_slice]):
                interval = True

        # Update arrays for diagnostics
        if (time_resolved or time_averaged or interval):
            if any(dict.get('N_e') for dict in self.master_diagnostic_dict.values()): self.update_N(self.species_names[0])
            if any(dict.get('N_i') for dict in self.master_diagnostic_dict.values()): self.update_N(self.species_names[1])
            if any(dict.get('W_e') for dict in self.master_diagnostic_dict.values()): self.update_W(self.species_names[0])
            if any(dict.get('W_i') for dict in self.master_diagnostic_dict.values()): self.update_W(self.species_names[1])
            if any(dict.get('E_z') for dict in self.master_diagnostic_dict.values()): self.update_E()
            if any(dict.get('phi') for dict in self.master_diagnostic_dict.values()): self.update_phi()
            if any(dict.get('Jze') for dict in self.master_diagnostic_dict.values()): self.update_Jz(self.species_names[0])
            if any(dict.get('Jzi') for dict in self.master_diagnostic_dict.values()): self.update_Jz(self.species_names[1])
            if any(dict.get('J_d') for dict in self.master_diagnostic_dict.values()): self.update_J_d()
            if any(dict.get('CPe') for dict in self.master_diagnostic_dict.values()): self.update_P_C(self.species_names[0])
            if any(dict.get('CPi') for dict in self.master_diagnostic_dict.values()): self.update_P_C(self.species_names[1])
            if any(dict.get('IPe') for dict in self.master_diagnostic_dict.values()): self.update_P_I(self.species_names[0])
            if any(dict.get('IPi') for dict in self.master_diagnostic_dict.values()): self.update_P_I(self.species_names[1])
            if any(dict.get('EEdf') for dict in self.master_diagnostic_dict.values()): self.calculate_eedf()
            if any(dict.get('IEdf') for dict in self.master_diagnostic_dict.values()): self.calculate_iedf(self.species_names[1])

        # Perform diagnostics
        if time_resolved:
            do_time_resolved_diagnostics(self.curr_tr)
            self.curr_tr += 1
        if time_averaged:
            do_time_averaged_diagnostics()
            self.curr_ta += 1
        if interval:
            do_interval_diagnostics(self.curr_slice)
            self.curr_slice += 1
            if self.curr_slice == len(self.in_coll_steps[self.curr_diag_output][self.curr_interval]):
                self.curr_interval += 1
                self.curr_slice = 0
                if self.curr_interval == len(self.in_coll_steps[self.curr_diag_output]):
                    self.curr_interval = 0

        # Save the electric field for the displacement current
        if save_E_last_step:
            E_wrapper = fields.EzFPWrapper()
            self.E_last_step = E_wrapper[...]

        # Finalize and save diagnostics
        if step == self.diag_stop[self.curr_diag_output]:

            # Save ieadf for each species and wall, if necessary
            if any(self.master_diagnostic_dict['ieadfs'].values()):
                for species in self.species_names[1:]:
                    for key, value in self.master_diagnostic_dict['ieadfs'].items():
                        if value:
                            self.calculate_ieadf(species, key)

            # Save ionization rate for each species, if necessary
            if self.Riz_switch and self.Riz_current_output <= self.Riz_max_output:
                self.add_bulk_Riz()
                self.add_buff_Riz()
                self.Riz_diag_counter += 1

            # Clear ieadf buffers
            self.clear_ieadf_buffers()

            # Finalize and save diagnostic data
            self.save_diagnostic_data()

            # Move to next diagnostic output
            self.curr_diag_output += 1
            self.curr_tr = 0
            self.curr_ta = 0
            self.curr_interval = 0
            self.curr_slice = 0

            # Reset diagnostic arrays
            if self.curr_diag_output < self.num_outputs:
                self.reset_diagnostic_arrays()

    def reset_diagnostic_arrays(self):
        '''
        Reset diagnostic arrays, call after the diagnostic output
        counter has been incremented.
        '''
        # Ieadf arrays
        self.ieadf_by_species = {}
        for species in self.species_names[1:]:
            self.ieadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['ieadfs'].items():
                if value:
                    self.ieadf_by_species[species][key] = np.zeros((len(self.ieadf_bin_centers), len(self.iadf_bin_centers)))

        # Ionization rate array
        if self.Riz_switch and self.Riz_diag_counter == 0:
            for species in self.species_names[1:]:
                self.Riz_by_species[species] = np.zeros((self.Riz_nt, self.nz))

        # Time resolved arrays
        self.tr_N_e = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_N_i = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_W_e = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_W_i = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_E_z = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_phi = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_Jze = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_Jzi = np.zeros((self.tr_coll[self.curr_diag_output], self.nz + 1))
        self.tr_J_d = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_CPe = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_CPi = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_IPe = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_IPi = np.zeros((self.tr_coll[self.curr_diag_output], self.nz))
        self.tr_EEdf = np.zeros((self.tr_coll[0], len(self.eedf_bin_centers)))
        self.tr_IEdf = np.zeros((self.tr_coll[0], len(self.iedf_bin_centers)))
        self.tr_times = np.zeros((self.tr_coll[self.curr_diag_output]))

        # Power arrays
        self.tr_Pin_vst = None
        self.tr_CPe_vst = None
        self.tr_CPi_vst = None
        self.tr_IPe_vst = None
        self.tr_IPi_vst = None

        # Time averaged arrays
        self.ta_N_e = np.zeros(self.nz + 1)
        self.ta_N_i = np.zeros(self.nz + 1)
        self.ta_W_e = np.zeros(self.nz + 1)
        self.ta_W_i = np.zeros(self.nz + 1)
        self.ta_E_z = np.zeros(self.nz)
        self.ta_phi = np.zeros(self.nz + 1)
        self.ta_Jze = np.zeros(self.nz + 1)
        self.ta_Jzi = np.zeros(self.nz + 1)
        self.ta_J_d = np.zeros(self.nz)
        self.ta_CPe = np.zeros(self.nz)
        self.ta_CPi = np.zeros(self.nz)
        self.ta_IPe = np.zeros(self.nz)
        self.ta_IPi = np.zeros(self.nz)
        self.ta_EEdf = np.zeros(len(self.eedf_bin_centers))
        self.ta_IEdf = np.zeros(len(self.iedf_bin_centers))

        # Interval arrays
        self.in_N_e = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_N_i = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_W_e = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_W_i = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_E_z = np.zeros((len(self.in_slices), self.nz))
        self.in_phi = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_Jze = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_Jzi = np.zeros((len(self.in_slices), self.nz + 1))
        self.in_J_d = np.zeros((len(self.in_slices), self.nz))
        self.in_CPe = np.zeros((len(self.in_slices), self.nz))
        self.in_CPi = np.zeros((len(self.in_slices), self.nz))
        self.in_IPe = np.zeros((len(self.in_slices), self.nz))
        self.in_IPi = np.zeros((len(self.in_slices), self.nz))
        self.in_EEdf = np.zeros((len(self.in_slices), len(self.eedf_bin_centers)))
        self.in_IEdf = np.zeros((len(self.in_slices), len(self.iedf_bin_centers)))

    ###########################################################################
    # Saving Functions                                                        #
    ###########################################################################
    def save_diagnostic_data(self):
        '''
        Save diagnostic data at the current time step
        '''
        def finalize_diagnostic_data():
            '''
            Finalize diagnostic data before saving
            '''
            # Grab species names
            species = self.species_names

            # Multiply by the time factor to get ionization rate
            if self.Riz_switch and self.Riz_diag_counter == 0:
                factor = 1.0 / (self.Riz_collection_time * self.dz)
                for spec in species[1:]:
                    self.Riz_by_species[spec] *= factor

            # Grab temporary dictionary for time resolved diagnostics
            active = self.master_diagnostic_dict['time_resolved']
            # Convert to correct units
            # if active['N_e']:
            #     self.tr_N_e /= self.charge_by_name[species[0]]
            # if active['N_i']:
            #     self.tr_N_i /= self.charge_by_name[species[1]]
            # Convert to correct units
            if active['N_e']:
                self.tr_N_e /= self.dz
            if active['N_i']:
                self.tr_N_i /= self.dz
            if active['W_e']:
                v2_factor = self.mass_by_name[species[0]] / 2.0 / constants.q_e
                self.tr_W_e *= v2_factor
            if active['W_i']:
                v2_factor = self.mass_by_name[species[1]] / 2.0 / constants.q_e
                self.tr_W_i *= v2_factor
            if active['Jze']:
                Jz_factor = self.charge_by_name[species[0]] / self.dz
                self.tr_Jze *= Jz_factor
            if active['Jzi']:
                Jz_factor = self.charge_by_name[species[1]] / self.dz
                self.tr_Jzi *= Jz_factor
            if active['J_d']:
                self.tr_J_d *= constants.ep0 / self.dt
            if active['CPe']:
                CP_factor = self.charge_by_name[species[0]] / self.dz
                self.tr_CPe *= CP_factor
            if active['CPi']:
                CP_factor = self.charge_by_name[species[1]] / self.dz
                self.tr_CPi *= CP_factor
            if active['IPe']:
                IP_factor = self.charge_by_name[species[0]] / self.dz
                self.tr_IPe *= IP_factor
            if active['IPi']:
                IP_factor = self.charge_by_name[species[1]] / self.dz
                self.tr_IPi *= IP_factor
            
            # Grab temporary dictionary for power diagnostics
            active = self.tr_power_dict
            if active['Pin_vst']:
                self.tr_Pin_vst = np.zeros(len(self.tr_times))
                for time_idx in range(len(self.tr_times)):
                    self.tr_Pin_vst[time_idx] = - (self.tr_Jzi[time_idx][-1] + self.tr_Jze[time_idx][-1]) * self.tr_phi[time_idx][-1]
            if active['CPe_vst'] or active['CPi_vst']:
                E_on_nodes = np.zeros_like(self.tr_phi)
                for time_idx in range(len(self.tr_times)):
                    E_on_nodes[time_idx] = -np.gradient(self.tr_phi[time_idx], self.dz)
            if active['CPe_vst']:
                self.tr_CPe_vst = np.zeros(len(self.tr_times))
                for time_idx in range(len(self.tr_times)):
                    self.tr_CPe_vst[time_idx] = np.trapz(E_on_nodes[time_idx] * self.tr_Jze[time_idx], self.nodes)
            if active['CPi_vst']:
                self.tr_CPi_vst = np.zeros(len(self.tr_times))
                for time_idx in range(len(self.tr_times)):
                    self.tr_CPi_vst[time_idx] = np.trapz(E_on_nodes[time_idx] * self.tr_Jzi[time_idx], self.nodes)
            if active['IPe_vst']:
                self.tr_IPe_vst = np.zeros(len(self.tr_times))
                for time_idx in range(len(self.tr_times)):
                    self.tr_IPe_vst[time_idx] = np.trapz(self.tr_IPe[time_idx], self.nodes)
            if active['IPi_vst']:
                self.tr_IPi_vst = np.zeros(len(self.tr_times))
                for time_idx in range(len(self.tr_times)):
                    self.tr_IPi_vst[time_idx] = np.trapz(self.tr_IPi[time_idx], self.nodes)
            
            # Grab temporary dictionary for time averaged diagnostics
            active = self.master_diagnostic_dict['time_averaged']
            collections = self.ta_coll[self.curr_diag_output]
            if active['N_e']:
                self.ta_N_e /= collections * self.dz
                # self.ta_N_e /= self.charge_by_name[species[0]]
            if active['N_i']:
                self.ta_N_i /= collections * self.dz
                # self.ta_N_i /= self.charge_by_name[species[1]]
            if active['W_e']:
                v2_factor = self.mass_by_name[species[0]] / 2.0 / constants.q_e
                self.ta_W_e *= v2_factor / collections
            if active['W_i']:
                v2_factor = self.mass_by_name[species[1]] / 2.0 / constants.q_e
                self.ta_W_i *= v2_factor / collections
            if active['E_z']:
                self.ta_E_z /= collections 
            if active['phi']:
                self.ta_phi /= collections
            if active['Jze']:
                Jz_factor = self.charge_by_name[species[0]] / self.dz
                self.ta_Jze *= Jz_factor / collections
            if active['Jzi']:
                Jz_factor = self.charge_by_name[species[1]] / self.dz
                self.ta_Jzi *= Jz_factor / collections
            if active['J_d']:
                Jd_factor = constants.ep0 / self.dt
                self.ta_J_d *= Jd_factor / collections
            if active['CPe']:
                CP_factor = self.charge_by_name[species[0]] / self.dz
                self.ta_CPe *= CP_factor / collections
            if active['CPi']:
                CP_factor = self.charge_by_name[species[1]] / self.dz
                self.ta_CPi *= CP_factor / collections
            if active['IPe']:
                IP_factor = self.charge_by_name[species[0]] / self.dz
                self.ta_IPe *= IP_factor / collections
            if active['IPi']:
                IP_factor = self.charge_by_name[species[1]] / self.dz
                self.ta_IPi *= IP_factor / collections
            if active['EEdf']:
                self.ta_EEdf /= collections
            if active['IEdf']:
                self.ta_IEdf /= collections
            
            # Grab temporary dictionary for interval diagnostics
            active = self.master_diagnostic_dict['interval']
            if active['N_e']:
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_N_e[ii] /= len(self.in_coll_steps[self.curr_diag_output]) * self.dz
                    # self.in_N_e[ii] /= self.charge_by_name[species[0]]
            if active['N_i']:
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_N_i[ii] /= len(self.in_coll_steps[self.curr_diag_output]) * self.dz
                    # self.in_N_i[ii] /= self.charge_by_name[species[1]]
            if active['W_e']:
                v2_factor = self.mass_by_name[species[0]] / 2.0 / constants.q_e
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_W_e[ii] *= v2_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['W_i']:
                v2_factor = self.mass_by_name[species[1]] / 2.0 / constants.q_e
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_W_i[ii] *= v2_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['E_z']:
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_E_z[ii] /= len(self.in_coll_steps[self.curr_diag_output])
            if active['phi']:
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_phi[ii] /= len(self.in_coll_steps[self.curr_diag_output])
            if active['Jze']:
                Jz_factor = self.charge_by_name[species[0]] / self.dz
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_Jze[ii] *= Jz_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['Jzi']:
                Jz_factor = self.charge_by_name[species[1]] / self.dz
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_Jzi[ii] *= Jz_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['J_d']:
                Jd_factor = constants.ep0 / self.dt
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_J_d[ii] *= Jd_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['IPe']:
                IP_factor = self.charge_by_name[species[0]] / self.dz
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_IPe[ii] *= IP_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['IPi']:
                IP_factor = self.charge_by_name[species[1]] / self.dz
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_IPi[ii] *= IP_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['CPe']:
                CP_factor = self.charge_by_name[species[0]] / self.dz
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_CPe[ii] *= CP_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['CPi']:
                CP_factor = self.charge_by_name[species[1]] / self.dz
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_CPi[ii] *= CP_factor / len(self.in_coll_steps[self.curr_diag_output])
            if active['EEdf']:
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_EEdf[ii] /= len(self.in_coll_steps[self.curr_diag_output])
            if active['IEdf']:
                for ii in range(len(self.in_slices)):
                    if len(self.in_coll_steps[self.curr_diag_output]) == 0:
                        continue
                    self.in_IEdf[ii] /= len(self.in_coll_steps[self.curr_diag_output])

        # Send ionization data to rank 0
        if self.Riz_switch and self.Riz_diag_counter == self.diag_collections_per_Riz:
            self.collect_Riz()

            # Reset counters
            self.Riz_diag_counter = 0
            self.Riz_current_output += 1

        if comm.rank != 0:
            return

        finalize_diagnostic_data()

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
        if any(self.master_diagnostic_dict['time_averaged'].values()) and not os.path.exists(ta_folder):
            os.makedirs(ta_folder)
        if any(self.master_diagnostic_dict['interval'].values()) and not os.path.exists(in_folder):
            os.makedirs(in_folder)

        # Save ieadfs
        active = self.master_diagnostic_dict['ieadfs']
        for key, val in active.items():
            if val:
                for species in self.species_names[1:]:
                    if key == 'z_lo':
                        prefix = 'lw'
                    elif key == 'z_hi':
                        prefix = 'rw'
                    np.save(os.path.join(self.ieadf_dir_by_species[species], f'{prefix}_{step:04d}.npy'), self.ieadf_by_species[species][key])

        # Save ionization rate histograms
        if self.Riz_switch and self.Riz_diag_counter == 0:
            # Save diagnostic
            if self.Riz_current_output - 1 <= self.Riz_max_output:
                for species in self.species_names[1:]:
                    np.save(os.path.join(self.Riz_dir_by_species[species], f'Riz_{step:04d}.npy'), self.Riz_by_species[species])

            # Turn off ionization rate diagnostic if we have reached the maximum number of outputs
            if self.Riz_current_output > self.Riz_max_output:
                self.Riz_switch = False

        # Save time resolved diagnostics
        active = self.master_diagnostic_dict['time_resolved']
        for key in active:
            if active[key]:
                np.save(os.path.join(tr_folder, f'{key}.npy'), getattr(self, f'tr_{key}'))
        if any(active.values()):
            np.save(os.path.join(tr_folder, 'times.npy'), self.tr_times)

        # Save time resolved power diagnostics
        active = self.tr_power_dict
        for key in active:
            if active[key]:
                np.save(os.path.join(tr_folder, f'{key}.npy'), getattr(self, f'tr_{key}'))

        # Save time averaged diagnostics
        active = self.master_diagnostic_dict['time_averaged']
        for key in active:
            if active[key]:
                np.save(os.path.join(ta_folder, f'{key}.npy'), getattr(self, f'ta_{key}'))

        # Save interval diagnostics
        active = self.master_diagnostic_dict['interval']
        if len(self.in_coll_steps[self.curr_diag_output]) > 0:
            for key in active:
                if active[key]:
                    arrays_dict = {f't{i+1:02d}': getattr(self, f'in_{key}')[i] for i in range(len(self.in_slices))}
                    np.savez(os.path.join(in_folder, f'{key}.npz'), **arrays_dict)

    ###########################################################################
    # Helper Functions                                                        #
    ###########################################################################
    def check_file(self, file_name):
        '''
        If the file exists, rename it to have '_old' before the extension.
        '''
        if os.path.exists(file_name):
            # Split the file name at the '.' and add '_old' before the extension
            file_name_split = file_name.split('.')
            file_name_split[-2] += '_old'
            old_file_name = '.'.join(file_name_split)
            os.rename(file_name, old_file_name)