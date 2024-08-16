from __future__ import annotations  # Allows using class names as type hints before they are fully defined
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inputs import CapacitiveDischargeExample  # Only imports Simulation for type checking to avoid circular import

import numpy as np
import sys, os

from pywarpx import fields, particle_containers, picmi
from mpi4py import MPI as mpi

# Initialize mpi communicator
comm = mpi.COMM_WORLD
num_proc = comm.Get_size()

constants = picmi.constants

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
        to initialize after installing all other diagnostics or
        checkpoints, since we initialize the inputs and warpx here.

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
                'IPe': False,
                'IPi': False,
                'J_d': False
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
                'IPe': False,
                'IPi': False,
                'J_d': True
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
                'IPe': False,
                'IPi': False,
                'J_d': False
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
                'IPe': False,
                'IPi': False,
                'J_d': False
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
                'IPe': False,
                'IPi': False,
                'J_d': True
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
                'IPe': False,
                'IPi': False,
                'J_d': False
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

        # Initialize simulation
        simulation_obj.sim.initialize_inputs()
        simulation_obj.sim.initialize_warpx()

        # Import simulation parameters
        self.m_ion = simulation_obj.m_ion
        self.rf_period = 1 / simulation_obj.freq
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

        if restart_checkpoint:
            self.convergence_time = self.sim_ext.warpx.gett_new(lev=0) + simulation_obj.convergence_time
            self.max_time = self.sim_ext.warpx.gett_new(lev=0) + simulation_obj.total_time
        else:
            self.convergence_time = simulation_obj.convergence_time
            self.max_time = simulation_obj.total_time
        self.diag_time = simulation_obj.diag_time
        self.evolve_time = simulation_obj.evolve_time

        self.num_in_tr = simulation_obj.collections_per_diag_step
        if self.num_in_tr > int(self.diag_time / self.dt):
            self.num_in_tr = int(self.diag_time / self.dt)

        self.in_period = self.rf_period
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
        self.original_interval_dict_array = [el for el in interval_dict.values()]

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
        self._get_diagnostic_steps()
        if any(interval_dict.values()):
            self._get_interval_collection_steps()
        else:
            self.output_next_in_coll = -1
        if self.Riz_switch:
            self._setup_Riz_diag(simulation_obj)
        self._setup_diagnostic_arrays(simulation_obj)

        # Save settings to file
        self._save_diagnostic_inputs()
        if self.master_diagnostic_dict['ieadfs']['z_lo'] or self.master_diagnostic_dict['ieadfs']['z_hi']:
            self._save_edf_settings()
        self._save_cells_and_nodes(simulation_obj)

        # If any IP diagnostics are on in time_averaged, time_resolved, or interval, get the icp field
        if any(dict.get('IPi') for dict in self.master_diagnostic_dict.values()) or any(dict.get('IPe') for dict in self.master_diagnostic_dict.values()):
            self.ICP_Ex_field = self._get_ICP_field(simulation_obj)

        # Set diagnostic output index
        self.curr_diag_output = 0

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
        self.iedf_bin_edges = np.linspace(0, simulation_obj.iedf_max_eV, simulation_obj.num_bins + 1)
        self.iedf_bin_centers = np.multiply(self.iedf_bin_edges[:-1] + self.iedf_bin_edges[1:], 0.5)
        self.iadf_bin_edges = np.linspace(-90, 90, 720 + 1)
        self.iadf_bin_centers = np.multiply(self.iadf_bin_edges[:-1] + self.iadf_bin_edges[1:], 0.5)

        # Ieadf arrays
        self.ieadf_by_species = {}
        for species in self.species_names[1:]:
            self.ieadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['ieadfs'].items():
                if value:
                    self.ieadf_by_species[species][key] = np.zeros((len(self.iedf_bin_centers), len(self.iadf_bin_centers)))

        # Ionization rate arrays
        if self.Riz_switch:
            self.Riz_by_species = {}
            for species in self.species_names[1:]:
                self.Riz_by_species[species] = np.zeros((self.Riz_nt, self.nz))

        # Time resolved arrays
        self.tr_N_e = None
        self.tr_N_i = None
        self.tr_W_e = None
        self.tr_W_i = None
        self.tr_E_z = None
        self.tr_phi = None
        self.tr_Jze = None
        self.tr_Jzi = None
        self.tr_IPe = None
        self.tr_IPi = None
        self.tr_J_d = None
        self.tr_times = None

        # Power arrays
        self.tr_Pin_vst = None
        self.tr_CPe_vst = None
        self.tr_CPi_vst = None
        self.tr_IPe_vst = None
        self.tr_IPi_vst = None

        # Time averaged arrays
        self.ta_N_e = None
        self.ta_N_i = None
        self.ta_W_e = None
        self.ta_W_i = None
        self.ta_E_z = None
        self.ta_phi = None
        self.ta_Jze = None
        self.ta_Jzi = None
        self.ta_IPe = None
        self.ta_IPi = None
        self.ta_J_d = None
        self.ta_times = None

        # Interval arrays
        self.in_N_e = [None for _ in range(len(self.in_slices))]
        self.in_N_i = [None for _ in range(len(self.in_slices))]
        self.in_W_e = [None for _ in range(len(self.in_slices))]
        self.in_W_i = [None for _ in range(len(self.in_slices))]
        self.in_E_z = [None for _ in range(len(self.in_slices))]
        self.in_phi = [None for _ in range(len(self.in_slices))]
        self.in_Jze = [None for _ in range(len(self.in_slices))]
        self.in_Jzi = [None for _ in range(len(self.in_slices))]
        self.in_IPe = [None for _ in range(len(self.in_slices))]
        self.in_IPi = [None for _ in range(len(self.in_slices))]
        self.in_J_d = [None for _ in range(len(self.in_slices))]

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
            self.J_d.append(np.zeros(self.nz + 1))
        self.J_d = np.stack(self.J_d)

        self.P = []
        for i in range(len(self.species_names)):
            self.P.append(np.zeros(self.nz + 1))
        self.P = np.stack(self.P)

        self.E = np.zeros(self.nz + 1)
        self.phi = np.zeros(self.nz + 1)
        self.E_last_step = np.zeros(self.nz + 1)
    
    def _setup_Riz_diag(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Set up diagnostics for ionization rate
        
        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
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

    def _get_ICP_field(self, simulation_obj: CapacitiveDischargeExample):
        '''
        Gets the ICP field from the input file

        Parameters
        ----------
        simulation_obj: CapacitiveDischargeExample
            Object of the main simulation class
        '''
        # Get ICP field min/max
        self.ICP_zmin = simulation_obj.ICP_zmin
        self.ICP_zmax = simulation_obj.ICP_zmax

        # Get the ICP field and frequency
        self.ICP_E_field = simulation_obj.ICP_E_field
        self.ICP_freq = simulation_obj.ICP_freq

        # Get the ICP field
        if simulation_obj.Ex_ext != 0:
            self.Ex_ext = simulation_obj.Ex_ext.replace('cos', 'np.cos').replace('sin', 'np.sin').replace('pi', 'np.pi')
        else:
            print('ERROR: ICP field in x direction is zero. Turn off ICP diagnostics.')
            sys.exit()
        if simulation_obj.Ey_ext != 0:
            print('ERROR: ICP field in y direction not supported')
            sys.exit()
        if simulation_obj.Ez_ext != 0:
            print('ERROR: ICP field in z direction not supported')
            sys.exit()
        
        def ICP_Ex_field(z: float):
            '''
            Returns the ICP field at a given z location at the current time.
            '''
            if z > self.ICP_zmin and z < self.ICP_zmax:
                return eval(self.Ex_ext, {'np': np, 't': self.sim_ext.warpx.gett_new(lev=0)})
            else:
                return 0.0
        
        return ICP_Ex_field

    def _get_interval_collection_steps(self):
        '''
        Get step numbers to perform interval diagnostics. Computes:
        - self.time_for_interval_collection: time for interval diagnostics
        '''
        # Get first set of collection times after convergence
        num_periods = self.convergence_time // self.in_period
        first_collection_times = (num_periods + self.in_slices) * self.in_period
        
        # Make sure collection steps are after convergence
        while all([int(t/self.dt) < self.diag_start[0] for t in first_collection_times]):
            first_collection_times += self.in_period
            self.step_for_in_collection = np.round(first_collection_times / self.dt).astype(int)

        # Convert times to steps
        self.step_for_in_collection = np.round(first_collection_times / self.dt).astype(int)

        # Make sure collection times fall within simulation
        for ii in range(len(self.step_for_in_collection) - 1, -1, -1):
            if self.step_for_in_collection[ii] > int(self.max_time / self.dt):
                # Remove the element
                self.step_for_in_collection = np.delete(self.step_for_in_collection, ii)
                self.in_slices = np.delete(self.in_slices, ii)
        # Turn off interval diagnostics if no times are left
        if len(self.step_for_in_collection) == 0:
            for key in self.master_diagnostic_dict['interval']:
                self.master_diagnostic_dict['interval'][key] = False
        else:
            # Find index of the next diagnostic output that each interval step fits into
            next_output = np.full_like(self.step_for_in_collection, -1)
            for ii in range(len(self.step_for_in_collection)):
                for jj in range(len(self.diag_stop)):
                    if (self.diag_start[jj] <= self.step_for_in_collection[ii] <= self.diag_stop[jj]):
                        next_output[ii] = jj
                        break
            # Save the next output
            try:
                self.output_next_in_coll = min(next_output[next_output != -1])
            except ValueError:
                self.output_next_in_coll = -1
            # Switch interval diagnostics off if necessary
            if self.output_next_in_coll > 0:
                for key in self.master_diagnostic_dict['interval']:
                    self.master_diagnostic_dict['interval'][key] = False 
            if self.output_next_in_coll == -1:
                # Set all of self.original_interval_dict to False to turn off interval diagnostics
                self.original_interval_dict_array = [False] * len(self.original_interval_dict_array)
                temp_dict = {key: bool for key, bool in zip(self.master_diagnostic_dict['interval'].keys(), self.original_interval_dict_array)}
                self.master_diagnostic_dict['interval'] = temp_dict

    def _get_diagnostic_steps(self):
        '''
        Get step numbers to perform diagnostics. Computes:
        - self.diag_start_step: first step for diagnostics
        - self.diag_stop_step: last step for diagnostics
        - self.diag_period_steps: number of steps between diagnostics
        - self.diag_time_resolving_steps: number of steps between time resolved
        diagnostic collections
        '''
        # Note: We calculate times in this function in seconds and then
        #       convert to time steps to get the most accurate step numbers
        diag_start = self.convergence_time
        diag_n_evolve = self.diag_time + self.evolve_time
        
        # Make a list of diagnostic start times
        diag_start_times = []
        for i in range(self.num_outputs):
            diag_start_times.append(diag_start + i * diag_n_evolve)
        diag_start_times = np.array(diag_start_times)

        # Get time between time resolved diagnostic collections
        self.tr_interval = self.diag_time / self.num_in_tr
        
        # Convert times to steps
        self.diag_start = np.round(diag_start_times / self.dt).astype(int)
        self.diag_period_steps = int(self.diag_time / self.dt)
        self.diag_stop = self.diag_start + self.diag_period_steps
        self.diag_time_resolving_steps = int(self.tr_interval / self.dt)

        # If diag_stop[ii] == diag_start[ii+1], shift diag_end[ii] back 1 step
        for ii in range(1, len(self.diag_start)):
            if self.diag_start[ii] == self.diag_stop[ii - 1]:
                self.diag_stop[ii - 1] -= 1

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
            f.write(f'Timestep [s]={self.dt:e}\n')
            f.write(f'Cell size [m]={self.dz:e}\n\n')

            f.write('Diagnostic Parameters\n')
            f.write('---------------------\n')
            f.write(f'Convergence time [s]={self.convergence_time}\n')
            f.write(f'Diagnostic time [s]={self.diag_time}\n')
            f.write(f'Evolve time [s]={self.evolve_time}\n\n')
            
            f.write(f'Number of diagnostic outputs={self.num_outputs}\n\n')

            f.write(f'Time [s] between time resolved collections={self.tr_interval}\n')
            f.write(f'Time resolved collections per diagnostic={self.num_in_tr}\n\n')

            f.write(f'Interval period [s]={self.in_period}\n')
            f.write(f'Times in interval={', '.join(map(str,self.in_slices))}\n\n')

            f.write(f'Output #   |   Start Step   |   Stop Step   |   Start Time   |   Start Time\n')
            f.write(f'------------------------------------------------------------------------------\n')
            for ii in range(self.num_outputs):
                f.write(f'   {ii+1:5d}   | {self.diag_start[ii]:12d}   |{self.diag_stop[ii]:12d}   | {self.diag_start[ii]*self.dt:.8e} | {self.diag_stop[ii]*self.dt:.8e}\n')

    def _save_edf_settings(self):
        '''
        Save the settings for energy distribution function creation
        '''
        if comm.rank != 0:
            return

        # Make a diagnostics directory
        if not os.path.exists(self.diag_folder):
            os.makedirs(self.diag_folder)

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
            np.save(f'{self.ieadf_dir_by_species[species]}/bins_eV.npy', self.iedf_bin_centers)
            np.save(f'{self.ieadf_dir_by_species[species]}/bins_deg.npy', self.iadf_bin_centers)
    
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

        # Sort by z and assign w to nodes
        temp_N = np.zeros(self.nz + 1)
        for ii in range(len(cell_idx)):
            # Get node
            idx = cell_idx[ii]

            if idx == self.nz:
                # If at the last node, just add to the last node
                temp_N[idx] += w[ii]
            else:
                # Weight particle to nodes
                weight_right = z[ii] / self.dz - idx
                weight_left = 1 - weight_right

                # Multiply by weight
                w_right = weight_right * w[ii]
                w_left = weight_left * w[ii]

                # Add particle to nodes ii and ii + 1
                temp_N[idx] += w_left
                temp_N[idx + 1] += w_right

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

        # Sort by z and assign v2 to nodes
        temp_W = np.zeros(self.nz + 1)
        temp_w = np.zeros(self.nz + 1)
        for ii in range(len(cell_idx)):
            # Get node
            idx = cell_idx[ii]

            if idx == self.nz:
                # If at the last node, just add to the last node
                temp_W[idx] += v2[ii] * w[ii]
                temp_w[idx] += w[ii]
            else:
                # Weight particle to nodes
                weight_right = z[ii] / self.dz - idx
                weight_left = 1 - weight_right

                # Multiply by weight (saves an operation to do this once)
                w_right = weight_right * w[ii]
                w_left = weight_left * w[ii]

                # Add particle to nodes ii and ii + 1
                temp_W[idx] += v2[ii] * w_left
                temp_w[idx] += w_left
                temp_W[idx + 1] += v2[ii] * w_right
                temp_w[idx + 1] += w_right

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

        # Sort by z and assign uz to nodes
        temp_J = np.zeros(self.nz + 1)
        for ii in range(len(cell_idx)):
            # Get node
            idx = cell_idx[ii]

            if idx == self.nz:
                # If at the last node, just add to the last node
                temp_J[idx] += uz[ii] * w[ii]
            else:
                # Weight particle to nodes
                weight_right = z[ii] / self.dz - idx
                weight_left = 1 - weight_right

                # Multiply by weight
                w_right = weight_right * w[ii]
                w_left = weight_left * w[ii]

                # Add particle to nodes ii and ii + 1
                temp_J[idx] += uz[ii] * w_left
                temp_J[idx + 1] += uz[ii] * w_right

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

    def update_ICP(self, species):
        '''
        Calculate power into plasma via an external ICP Field.
        Needs to be multiplied by charge and divided by cell size before
        being used.
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

        # Get cell index of particle
        cell_idx = np.floor(z / self.dz).astype(int)

        # Get the ICP field at the particle locations (as currently is, the field is constant when it is turned on)
        curr_ICP_field = self.ICP_Ex_field(0.5 * (self.ICP_zmin + self.ICP_zmax))
        ICP_Ex = np.zeros(len(z))
        for ii in range(len(z)):
            if z[ii] > self.ICP_zmin and z[ii] < self.ICP_zmax:
                ICP_Ex[ii] = curr_ICP_field
            else:
                ICP_Ex[ii] = 0.0

        # Calculate the power into the plasma
        Pow = np.multiply(ux, ICP_Ex)

        # Sort by z and assign P to nodes
        temp_P = np.zeros(self.nz + 1)
        for ii in range(len(cell_idx)):
            # Get node
            idx = cell_idx[ii]

            if idx == self.nz:
                # If at the last node, just add to the last node
                temp_P[idx] += Pow[ii] * w[ii]
            else:
                # Weight particle to nodes
                weight_right = z[ii] / self.dz - idx
                weight_left = 1 - weight_right

                # Multiply by weight
                w_right = weight_right * w[ii]
                w_left = weight_left * w[ii]

                # Add particle to nodes ii and ii + 1
                temp_P[idx] += Pow[ii] * w_left
                temp_P[idx + 1] += Pow[ii] * w_right

        # Note: We don't need to synchronize if all processes have particles
        #       that are in the same cells... The next few lines may be worth
        #       adjusting later on.

        # Send temp_P to all processes
        P_data = np.zeros_like(temp_P)
        comm.Allreduce(temp_P, P_data, op=mpi.SUM)

        # Get index of species array in stack
        idx = self.diag_idx_by_name[species]

        # Report the temperature
        self.P[idx] = P_data

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
                return np.zeros((len(self.iedf_bin_centers), len(self.iadf_bin_centers)))

            # Calculate the ion energy and base its sign on the z velocity, but if z velocity is zero, use x velocity
            v2 = (np.square(ux) + np.square(uy) + np.square(uz))
            E = np.multiply(v2, 0.5 * self.m_ion / constants.q_e)

            # Calculate the ion xy velocity
            vxy = np.sqrt(np.square(ux) + np.square(uy))
            # Calculate angle with a negative sign so that left/right wall ieadfs are on the left/right of an energy vs angle plot
            angle = np.arctan(vxy / uz) * 180 / np.pi

            # Get the histogram (unnormalized)
            hist, *_ = np.histogram2d(E, angle, bins=[self.iedf_bin_edges, self.iadf_bin_edges], density=False, weights=w/self.dz)

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
        boundary_wrapper = particle_containers.ParticleBoundaryBufferWrapper()
        boundary_wrapper.clear_buffer()

    def calculate_Riz(self):
        '''
        Calculate the ionization rate for each species. Make sure to call this
        function before clear_ieadf_buffers() to ensure that the ionization
        rate considers any particles that exited the system.

        '''
        for spec in self.species_names[1:]:
            # Set up wrappers
            bd_wrapper = particle_containers.ParticleBoundaryBufferWrapper()
            sp_wrapper = particle_containers.ParticleContainerWrapper(spec)

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

            # Get bulk particle data
            try:
                sp_t = np.concatenate(sp_wrapper.get_particle_real_arrays('orig_t', 0))
                sp_z = np.concatenate(sp_wrapper.get_particle_real_arrays('orig_z', 0))
                sp_w = np.concatenate(sp_wrapper.get_particle_weight())
            except ValueError:
                sp_t = np.array([])
                sp_z = np.array([])
                sp_w = np.array([])

            # Make the current time the end of the time window
            t_end = self.sim_ext.warpx.gett_new(lev=0)
            t_begin = t_end - self.diag_time

            # Now, make a histogram of the ionization rate
            for z, t, w in zip(np.concatenate([bd_z['z_lo'], bd_z['z_hi'], sp_z]), np.concatenate([bd_t['z_lo'], bd_t['z_hi'], sp_t]), np.concatenate([bd_w['z_lo'], bd_w['z_hi'], sp_w])):
                # Check if the particle is in the time window
                if t < t_begin:
                    continue

                # Get the cell index
                z_idx = int(z / self.dz)

                # Get the time index
                t_idx = int(t / self.dt) % self.Riz_nt

                # Increment the ionization rate
                self.Riz_by_species[spec][t_idx][z_idx] += w

            # Sum the ionization rate histograms from all processors
            Riz_all = np.zeros_like(self.Riz_by_species[spec])
            comm.Allreduce(self.Riz_by_species[spec], Riz_all, op=mpi.SUM)
            self.Riz_by_species[spec] = Riz_all

    ###########################################################################
    # Simulation Functions                                                    #
    ###########################################################################
    def do_diagnostics(self):
        '''
        Master function to perform diagnostics at each time step. Should be
        installed at least one step before the first diagnostic step.
        '''        
        def do_time_resolved_diagnostics():
            '''
            Performs time resolved diagnostics
            '''
            # Grab temporary dictionary for time resolved diagnostics
            temp_settings = self.master_diagnostic_dict['time_resolved']

            if temp_settings['N_e']:
                if self.tr_N_e is None:
                    self.tr_N_e = self.N[0]
                else:
                    self.tr_N_e = np.vstack((self.tr_N_e, self.N[0]))
            if temp_settings['N_i']:
                if self.tr_N_i is None:
                    self.tr_N_i = self.N[1]
                else:
                    self.tr_N_i = np.vstack((self.tr_N_i, self.N[1]))
            if temp_settings['W_e']:
                if self.tr_W_e is None:
                    self.tr_W_e = self.W[0]
                else:
                    self.tr_W_e = np.vstack((self.tr_W_e, self.W[0]))
            if temp_settings['W_i']:
                if self.tr_W_i is None:
                    self.tr_W_i = self.W[1]
                else:
                    self.tr_W_i = np.vstack((self.tr_W_i, self.W[1]))
            if temp_settings['E_z']:
                if self.tr_E_z is None:
                    self.tr_E_z = self.E
                else:
                    self.tr_E_z = np.vstack((self.tr_E_z, self.E))
            if temp_settings['phi']:
                if self.tr_phi is None:
                    self.tr_phi = self.phi
                else:
                    self.tr_phi = np.vstack((self.tr_phi, self.phi))
            if temp_settings['Jze']:
                if self.tr_Jze is None:
                    self.tr_Jze = self.J[0]
                else:
                    self.tr_Jze = np.vstack((self.tr_Jze, self.J[0]))
            if temp_settings['Jzi']:
                if self.tr_Jzi is None:
                    self.tr_Jzi = self.J[1]
                else:
                    self.tr_Jzi = np.vstack((self.tr_Jzi, self.J[1]))
            if temp_settings['IPe']:
                if self.tr_IPe is None:
                    self.tr_IPe = self.P[0]
                else:
                    self.tr_IPe = np.vstack((self.tr_IPe, self.P[0]))
            if temp_settings['IPi']:
                if self.tr_IPi is None:
                    self.tr_IPi = self.P[1]
                else:
                    self.tr_IPi = np.vstack((self.tr_IPi, self.P[1]))
            if temp_settings['J_d']:
                if self.tr_J_d is None:
                    self.tr_J_d = self.J_d
                else:
                    self.tr_J_d = np.vstack((self.tr_J_d, self.J_d))
            
            # Append the time to the array
            if self.tr_times is None:
                self.tr_times = np.array([self.sim_ext.warpx.gett_new(lev=0)])
            else:
                self.tr_times = np.append(self.tr_times, self.sim_ext.warpx.gett_new(lev=0))

        def do_time_averaged_diagnostics():
            '''
            Performs time averaged diagnostics
            '''
            # Grab temporary dictionary for time averaged diagnostics
            temp_settings = self.master_diagnostic_dict['time_averaged']

            if temp_settings['N_e']:
                if self.ta_N_e is None:
                    self.ta_N_e = self.N[0]
                else:
                    self.ta_N_e = np.vstack((self.ta_N_e, self.N[0]))
            if temp_settings['N_i']:
                if self.ta_N_i is None:
                    self.ta_N_i = self.N[1]
                else:
                    self.ta_N_i = np.vstack((self.ta_N_i, self.N[1]))
            if temp_settings['W_e']:
                if self.ta_W_e is None:
                    self.ta_W_e = self.W[0]
                else:
                    self.ta_W_e = np.vstack((self.ta_W_e, self.W[0]))
            if temp_settings['W_i']:
                if self.ta_W_i is None:
                    self.ta_W_i = self.W[1]
                else:
                    self.ta_W_i = np.vstack((self.ta_W_i, self.W[1]))
            if temp_settings['E_z']:
                if self.ta_E_z is None:
                    self.ta_E_z = self.E
                else:
                    self.ta_E_z = np.vstack((self.ta_E_z, self.E))
            if temp_settings['phi']:
                if self.ta_phi is None:
                    self.ta_phi = self.phi
                else:
                    self.ta_phi = np.vstack((self.ta_phi, self.phi))
            if temp_settings['Jze']:
                if self.ta_Jze is None:
                    self.ta_Jze = self.J[0]
                else:
                    self.ta_Jze = np.vstack((self.ta_Jze, self.J[0]))
            if temp_settings['Jzi']:
                if self.ta_Jzi is None:
                    self.ta_Jzi = self.J[1]
                else:
                    self.ta_Jzi = np.vstack((self.ta_Jzi, self.J[1]))
            if temp_settings['IPe']:
                if self.ta_IPe is None:
                    self.ta_IPe = self.P[0]
                else:
                    self.ta_IPe = np.vstack((self.ta_IPe, self.P[0]))
            if temp_settings['IPi']:
                if self.ta_IPi is None:
                    self.ta_IPi = self.P[1]
                else:
                    self.ta_IPi = np.vstack((self.ta_IPi, self.P[1]))
            if temp_settings['J_d']:
                if self.ta_J_d is None:
                    self.ta_J_d = self.J_d
                else:
                    self.ta_J_d = np.vstack((self.ta_J_d, self.J_d))

        def do_interval_diagnostics(interval_idx):
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

            if temp_settings['N_e']:
                if self.in_N_e[interval_idx] is None:
                    self.in_N_e[interval_idx] = self.N[0]
                else:
                    self.in_N_e[interval_idx] = np.vstack((self.in_N_e[interval_idx], self.N[0]))
            if temp_settings['N_i']:
                if self.in_N_i[interval_idx] is None:
                    self.in_N_i[interval_idx] = self.N[1]
                else:
                    self.in_N_i[interval_idx] = np.vstack((self.in_N_i[interval_idx], self.N[1]))
            if temp_settings['W_e']:
                if self.in_W_e[interval_idx] is None:
                    self.in_W_e[interval_idx] = self.W[0]
                else:
                    self.in_W_e[interval_idx] = np.vstack((self.in_W_e[interval_idx], self.W[0]))
            if temp_settings['W_i']:
                if self.in_W_i[interval_idx] is None:
                    self.in_W_i[interval_idx] = self.W[1]
                else:
                    self.in_W_i[interval_idx] = np.vstack((self.in_W_i[interval_idx], self.W[1]))
            if temp_settings['E_z']:
                if self.in_E_z[interval_idx] is None:
                    self.in_E_z[interval_idx] = self.E
                else:
                    self.in_E_z[interval_idx] = np.vstack((self.in_E_z[interval_idx], self.E))
            if temp_settings['phi']:
                if self.in_phi[interval_idx] is None:
                    self.in_phi[interval_idx] = self.phi
                else:
                    self.in_phi[interval_idx] = np.vstack((self.in_phi[interval_idx], self.phi))
            if temp_settings['Jze']:
                if self.in_Jze[interval_idx] is None:
                    self.in_Jze[interval_idx] = self.J[0]
                else:
                    self.in_Jze[interval_idx] = np.vstack((self.in_Jze[interval_idx], self.J[0]))
            if temp_settings['Jzi']:
                if self.in_Jzi[interval_idx] is None:
                    self.in_Jzi[interval_idx] = self.J[1]
                else:
                    self.in_Jzi[interval_idx] = np.vstack((self.in_Jzi[interval_idx], self.J[1]))
            if temp_settings['IPe']:
                if self.in_IPe[interval_idx] is None:
                    self.in_IPe[interval_idx] = self.P[0]
                else:
                    self.in_IPe[interval_idx] = np.vstack((self.in_IPe[interval_idx], self.P[0]))
            if temp_settings['IPi']:
                if self.in_IPi[interval_idx] is None:
                    self.in_IPi[interval_idx] = self.P[1]
                else:
                    self.in_IPi[interval_idx] = np.vstack((self.in_IPi[interval_idx], self.P[1]))
            if temp_settings['J_d']:
                if self.in_J_d[interval_idx] is None:
                    self.in_J_d[interval_idx] = self.J_d
                else:
                    self.in_J_d[interval_idx] = np.vstack((self.in_J_d[interval_idx], self.J_d))

            # Add one to the number of collections for this interval
            self.num_in_collections_this_output[interval_idx] += 1

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

            # Prepare interval diagnostic settings for this collection
            if any(self.original_interval_dict_array):
                if self.curr_diag_output == self.output_next_in_coll:
                    # Revert interval diagnostics back to original settings
                    temp_dict = {key: bool for key, bool in zip(self.master_diagnostic_dict['interval'].keys(), self.original_interval_dict_array)}
                    self.master_diagnostic_dict['interval'] = temp_dict
            self.num_in_collections_this_output = [0 for _ in range(len(self.in_slices))]

        # Check if we need to save the electric field for the displacement current
        save_E_last_step = False
        if self.master_diagnostic_dict['time_resolved']['J_d'] and (next_step - self.diag_start[self.curr_diag_output]) % self.diag_time_resolving_steps == 0:
            save_E_last_step = True
        if self.master_diagnostic_dict['time_averaged']['J_d'] and (next_step >= self.diag_start[self.curr_diag_output]):
            save_E_last_step = True
        if self.master_diagnostic_dict['interval']['J_d'] and (next_step in self.step_for_in_collection):
            save_E_last_step = True

        # Go through each diagnostic type and determine if we need to update
        # arrays for that diagnostic at this time step
        time_resolved = False
        time_averaged = False
        interval = False
        turn_off_in = False
        if any(self.master_diagnostic_dict['time_resolved'].values()) and ((step - self.diag_start[self.curr_diag_output]) % self.diag_time_resolving_steps == 0) and step >= self.diag_start[self.curr_diag_output]:
            time_resolved = True
        if any(self.master_diagnostic_dict['time_averaged'].values()) and (step >= self.diag_start[self.curr_diag_output]):
            time_averaged = True
        if any(self.master_diagnostic_dict['interval'].values()) and (step in self.step_for_in_collection):
            interval = True
            interval_idx = np.where(self.step_for_in_collection == step)[0][0]
            # Calculate the next time for interval collections, if necessary
            if step == self.step_for_in_collection[-1]:
                # Update for next collection times
                current_time = self.sim_ext.warpx.gett_new(lev=0)

                num_periods = current_time // self.in_period
                next_collection_time = (num_periods + self.in_slices) * self.in_period

                while int(next_collection_time[-1] / self.dt) <= step:
                    next_collection_time += self.in_period

                # Convert times to steps
                self.step_for_in_collection = np.round(next_collection_time / self.dt).astype(int)

                # Find the index of the next diagnostic output that each interval step occurs during
                next_output = np.full_like(self.step_for_in_collection, -1)
                for ii in range(len(self.step_for_in_collection)):
                    for jj in range(len(self.diag_stop)):
                        if (self.diag_start[jj] <= self.step_for_in_collection[ii] <= self.diag_stop[jj]):
                            next_output[ii] = jj
                            break

                if all(next_output == -1):
                    if self.curr_diag_output == self.num_outputs - 1:
                        turn_off_in = True
                    else:
                        while int(next_collection_time[-1] / self.dt) < self.diag_start[self.curr_diag_output + 1]:
                            next_collection_time += self.in_period
                        
                        # Convert times to steps
                        self.step_for_in_collection = np.round(next_collection_time / self.dt).astype(int)

                        # Find the index of the next diagnostic output that each interval step occurs during
                        next_output = np.full_like(self.step_for_in_collection, -1)
                        for ii in range(len(self.step_for_in_collection)):
                            for jj in range(len(self.diag_stop)):
                                if (self.diag_start[jj] <= self.step_for_in_collection[ii] <= self.diag_stop[jj]):
                                    next_output[ii] = jj
                                    break
                
                if self.curr_diag_output not in next_output:
                    turn_off_in = True
                # Set the next output to the next output in the collection, and make sure it is not equal to -1
                if self.curr_diag_output == self.num_outputs - 1:
                    try:
                        self.output_next_in_coll = min(next_output[next_output != -1])
                    except ValueError:
                        # This means that all of the collection steps fall after the last diagnostic stop step
                        turn_off_in = True
                else:
                    self.output_next_in_coll = min(next_output[next_output != -1])

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
            if any(dict.get('IPe') for dict in self.master_diagnostic_dict.values()): self.update_ICP(self.species_names[0])
            if any(dict.get('IPi') for dict in self.master_diagnostic_dict.values()): self.update_ICP(self.species_names[1])
            if any(dict.get('J_d') for dict in self.master_diagnostic_dict.values()): self.update_J_d()

        # Perform diagnostics
        if time_resolved:
            do_time_resolved_diagnostics()
        if time_averaged:
            do_time_averaged_diagnostics()
        if interval:
            do_interval_diagnostics(interval_idx)

        # Save the electric field for the displacement current
        if save_E_last_step:
            E_wrapper = fields.EzFPWrapper()
            self.E_last_step = E_wrapper[...]

        # Finalize and save diagnostics
        if step == self.diag_stop[self.curr_diag_output]:

            reset_at_end = False
            if not all(self.master_diagnostic_dict['interval'].values()) and any(self.original_interval_dict_array):
                # Set self.master_diagnostic_dict['interval'] back to original settings for last step
                temp_dict = {key: bool for key, bool in zip(self.master_diagnostic_dict['interval'].keys(), self.original_interval_dict_array)}
                self.master_diagnostic_dict['interval'] = temp_dict
                reset_at_end = True

            # Save ieadf for each species and wall, if necessary
            if any(self.master_diagnostic_dict['ieadfs'].values()):
                for species in self.species_names[1:]:
                    for key, value in self.master_diagnostic_dict['ieadfs'].items():
                        if value:
                            self.calculate_ieadf(species, key)

            # Save ionization rate histogram for each species, if necessary
            if self.Riz_switch:
                self.calculate_Riz()

            # Clear ieadf buffers
            self.clear_ieadf_buffers()

            # Finalize and save diagnostic data
            self.save_diagnostic_data()

            # Reset diagnostic arrays
            self.reset_diagnostic_arrays()

            # Move to next diagnostic output
            self.curr_diag_output += 1

            # If necessary, update the next interval diagnostic collection step
            if any(self.master_diagnostic_dict['interval'].values()) and (self.curr_diag_output < self.num_outputs):
                # Find the next diagnostic output that each interval collection step fits into
                next_output = np.full_like(self.step_for_in_collection, -1)
                for ii in range(len(self.step_for_in_collection)):
                    for jj in range(len(self.diag_stop)):
                        if (self.diag_start[jj] <= self.step_for_in_collection[ii] <= self.diag_stop[jj]):
                            next_output[ii] = jj
                            break

                # If none of the collection steps fall into a diagnostic output, find the next valid output
                if any(next_output == -1):
                    # If all of the collection steps fall before the next diagnostic start step
                    if all(self.step_for_in_collection < self.diag_start[self.curr_diag_output]):

                        # Get a new set of collection times
                        current_time = self.sim_ext.warpx.gett_new(lev=0)

                        num_periods = current_time // self.in_period
                        next_collection_time = (num_periods + self.in_slices) * self.in_period

                        while int(next_collection_time[-1] / self.dt) < self.diag_start[self.curr_diag_output]:
                            next_collection_time += self.in_period

                        # Convert times to steps
                        self.step_for_in_collection = np.round(next_collection_time / self.dt).astype(int)

                        # Find the index of the next diagnostic output that each interval step occurs during
                        next_output = np.full_like(self.step_for_in_collection, -1)
                        for ii in range(len(self.step_for_in_collection)):
                            for jj in range(len(self.diag_stop)):
                                if (self.diag_start[jj] <= self.step_for_in_collection[ii] <= self.diag_stop[jj]):
                                    next_output[ii] = jj
                                    break

                    # If all of the collection steps fall after the last diagnostic stop step
                    if all(self.step_for_in_collection > self.diag_stop[-1]):
                        turn_off_in = True

                if self.curr_diag_output not in next_output:
                    turn_off_in = True
                # Save the next diagnostic output that the next interval output is saved at
                self.output_next_in_coll = min(next_output[next_output != -1])

            if reset_at_end:
                # Turn interval diagnostics off for the rest of the diagnostic interval
                for key in self.master_diagnostic_dict['interval']:
                    self.master_diagnostic_dict['interval'][key] = False

        if turn_off_in:
            # Turn interval diagnostics off for the rest of the diagnostic interval
            for key in self.master_diagnostic_dict['interval']:
                self.master_diagnostic_dict['interval'][key] = False

    def reset_diagnostic_arrays(self):
        '''
        Reset diagnostic arrays
        '''
        # Ieadf arrays
        self.ieadf_by_species = {}
        for species in self.species_names[1:]:
            self.ieadf_by_species[species] = {}
            # Create arrays for z_lo and z_hi, if they are turned on
            for key, value in self.master_diagnostic_dict['ieadfs'].items():
                if value:
                    self.ieadf_by_species[species][key] = np.zeros((len(self.iedf_bin_centers), len(self.iadf_bin_centers)))

        # Ionization rate array
        if self.Riz_switch:
            for species in self.species_names[1:]:
                self.Riz_by_species[species] = np.zeros((self.Riz_nt, self.nz))

        # Time resolved arrays
        self.tr_N_e = None
        self.tr_N_i = None
        self.tr_W_e = None
        self.tr_W_i = None
        self.tr_E_z = None
        self.tr_phi = None
        self.tr_Jze = None
        self.tr_Jzi = None
        self.tr_IPe = None
        self.tr_IPi = None
        self.tr_J_d = None
        self.tr_times = None

        # Power arrays
        self.tr_Pin_vst = None
        self.tr_CPe_vst = None
        self.tr_CPi_vst = None
        self.tr_IPe_vst = None
        self.tr_IPi_vst = None

        # Time averaged arrays
        self.ta_N_e = None
        self.ta_N_i = None
        self.ta_W_e = None
        self.ta_W_i = None
        self.ta_E_z = None
        self.ta_phi = None
        self.ta_Jze = None
        self.ta_Jzi = None
        self.ta_IPe = None
        self.ta_IPi = None
        self.ta_J_d = None
        self.ta_times = None

        # Interval arrays
        self.in_N_e = [None for _ in range(len(self.in_slices))]
        self.in_N_i = [None for _ in range(len(self.in_slices))]
        self.in_W_e = [None for _ in range(len(self.in_slices))]
        self.in_W_i = [None for _ in range(len(self.in_slices))]
        self.in_E_z = [None for _ in range(len(self.in_slices))]
        self.in_phi = [None for _ in range(len(self.in_slices))]
        self.in_Jze = [None for _ in range(len(self.in_slices))]
        self.in_Jzi = [None for _ in range(len(self.in_slices))]
        self.in_IPe = [None for _ in range(len(self.in_slices))]
        self.in_IPi = [None for _ in range(len(self.in_slices))]
        self.in_J_d = [None for _ in range(len(self.in_slices))]

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
            if self.Riz_switch:
                factor = 1.0 / (self.diag_time * self.dz)
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
            if active['IPe']:
                IP_factor = self.charge_by_name[species[0]] / self.dz
                self.tr_IPe *= IP_factor
            if active['IPi']:
                IP_factor = self.charge_by_name[species[1]] / self.dz
                self.tr_IPi *= IP_factor
            if active['J_d']:
                self.tr_J_d *= constants.ep0 / self.dt
            
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
            if active['N_e']:
                collections = len(self.ta_N_e)
                self.ta_N_e = np.sum(self.ta_N_e, axis=0) / collections # Average
                self.ta_N_e /= self.dz
                # self.ta_N_e /= self.charge_by_name[species[0]]
            if active['N_i']:
                collections = len(self.ta_N_i)
                self.ta_N_i = np.sum(self.ta_N_i, axis=0) / collections # Average
                self.ta_N_i /= self.dz
                # self.ta_N_i /= self.charge_by_name[species[1]]
            if active['W_e']:
                collections = len(self.ta_W_e)
                self.ta_W_e = np.sum(self.ta_W_e, axis=0) / collections # Average
                v2_factor = self.mass_by_name[species[0]] / 2.0 / constants.q_e
                self.ta_W_e *= v2_factor
            if active['W_i']:
                collections = len(self.ta_W_i)
                self.ta_W_i = np.sum(self.ta_W_i, axis=0) / collections # Average
                v2_factor = self.mass_by_name[species[1]] / 2.0 / constants.q_e
                self.ta_W_i *= v2_factor
            if active['E_z']:
                collections = len(self.ta_E_z)
                self.ta_E_z = np.sum(self.ta_E_z, axis=0) / collections 
            if active['phi']:
                collections = len(self.ta_phi)
                self.ta_phi = np.sum(self.ta_phi, axis=0) / collections
            if active['Jze']:
                collections = len(self.ta_Jze)
                self.ta_Jze = np.sum(self.ta_Jze, axis=0) / collections
                Jz_factor = self.charge_by_name[species[0]] / self.dz
                self.ta_Jze *= Jz_factor
            if active['Jzi']:
                collections = len(self.ta_Jzi)
                self.ta_Jzi = np.sum(self.ta_Jzi, axis=0) / collections
                Jz_factor = self.charge_by_name[species[1]] / self.dz
                self.ta_Jzi *= Jz_factor
            if active['IPe']:
                collections = len(self.ta_IPe)
                self.ta_IPe = np.sum(self.ta_IPe, axis=0) / collections
                IP_factor = self.charge_by_name[species[0]] / self.dz
                self.ta_IPe *= IP_factor
            if active['IPi']:
                collections = len(self.ta_IPi)
                self.ta_IPi = np.sum(self.ta_IPi, axis=0) / collections
                IP_factor = self.charge_by_name[species[1]] / self.dz
                self.ta_IPi *= IP_factor
            if active['J_d']:
                collections = len(self.ta_J_d)
                self.ta_J_d = np.sum(self.ta_J_d, axis=0) / collections
                self.ta_J_d *= constants.ep0 / self.dt
            
            # Grab temporary dictionary for interval diagnostics
            active = self.master_diagnostic_dict['interval']
            if active['N_e']:
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_N_e = np.array(self.in_N_e[0]) / self.dz # / self.charge_by_name[species[0]]
                    else:
                        collections = len(self.in_N_e)
                        self.in_N_e = np.sum(self.in_N_e, axis=0) / collections
                        self.in_N_e /= self.dz
                        # self.in_N_e /= self.charge_by_name[species[0]]
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_N_e[ii] = np.array(self.in_N_e[ii]) / self.dz # / self.charge_by_name[species[0]]
                        else:
                            self.in_N_e[ii] = np.sum(self.in_N_e[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_N_e[ii] /= self.dz
                            # self.in_N_e[ii] /= self.charge_by_name[species[0]]
            if active['N_i']:
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_N_i = np.array(self.in_N_i[1]) / self.dz # / self.charge_by_name[species[1]]
                    else:
                        collections = len(self.in_N_i)
                        self.in_N_i = np.sum(self.in_N_i, axis=0) / collections
                        self.in_N_i /= self.dz
                        # self.in_N_i /= self.charge_by_name[species[1]]
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_N_i[ii] = np.array(self.in_N_i[ii]) / self.dz # / self.charge_by_name[species[1]]
                        else:
                            self.in_N_i[ii] = np.sum(self.in_N_i[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_N_i[ii] /= self.dz
                            # self.in_N_i[ii] /= self.charge_by_name[species[1]]
            if active['W_e']:
                v2_factor = self.mass_by_name[species[0]] / 2.0 / constants.q_e
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_W_e = np.array(self.in_W_e[1]) * v2_factor
                    else:
                        collections = len(self.in_W_e)
                        self.in_W_e = np.sum(self.in_W_e, axis=0) / collections
                        self.in_W_e *= v2_factor
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_W_e[ii] = np.array(self.in_W_e[ii]) * v2_factor
                        else:
                            self.in_W_e[ii] = np.sum(self.in_W_e[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_W_e[ii] *= v2_factor
            if active['W_i']:
                v2_factor = self.mass_by_name[species[1]] / 2.0 / constants.q_e
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_W_i = np.array(self.in_W_i[1]) * v2_factor
                    else:
                        collections = len(self.in_W_i)
                        self.in_W_i = np.sum(self.in_W_i, axis=0) / collections
                        self.in_W_i *= v2_factor
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_W_i[ii] = np.array(self.in_W_i[ii]) * v2_factor
                        else:
                            self.in_W_i[ii] = np.sum(self.in_W_i[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_W_i[ii] *= v2_factor
            if active['E_z']:
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_E_z = np.array(self.in_E_z[1])
                    else:
                        collections = len(self.in_E_z)
                        self.in_E_z = np.sum(self.in_E_z, axis=0) / collections
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_E_z[ii] = np.array(self.in_E_z[ii])
                        else:
                            self.in_E_z[ii] = np.sum(self.in_E_z[ii], axis=0) / self.num_in_collections_this_output[ii]
            if active['phi']:
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_phi = np.array(self.in_phi[1])
                    else:
                        collections = len(self.in_phi)
                        self.in_phi = np.sum(self.in_phi, axis=0) / collections
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_phi[ii] = np.array(self.in_phi[ii])
                        else:
                            self.in_phi[ii] = np.sum(self.in_phi[ii], axis=0) / self.num_in_collections_this_output[ii]
            if active['Jze']:
                Jz_factor = self.charge_by_name[species[0]] / self.dz
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_Jze = np.array(self.in_Jze[1]) * Jz_factor
                    else:
                        collections = len(self.in_Jze)
                        self.in_Jze = np.sum(self.in_Jze, axis=0) / collections
                        self.in_Jze *= Jz_factor
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_Jze[ii] = np.array(self.in_Jze[ii]) * Jz_factor
                        else:
                            self.in_Jze[ii] = np.sum(self.in_Jze[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_Jze[ii] *= Jz_factor
            if active['Jzi']:
                Jz_factor = self.charge_by_name[species[1]] / self.dz
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_Jzi = np.array(self.in_Jzi[1]) * Jz_factor
                    else:
                        collections = len(self.in_Jzi)
                        self.in_Jzi = np.sum(self.in_Jzi, axis=0) / collections
                        self.in_Jzi *= Jz_factor
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_Jzi[ii] = np.array(self.in_Jzi[ii]) * Jz_factor
                        else:
                            self.in_Jzi[ii] = np.sum(self.in_Jzi[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_Jzi[ii] *= Jz_factor
            if active['IPe']:
                IP_factor = self.charge_by_name[species[0]] / self.dz
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_IPe = np.array(self.in_IPe[1]) * IP_factor
                    else:
                        collections = len(self.in_IPe)
                        self.in_IPe = np.sum(self.in_IPe, axis=0) / collections
                        self.in_IPe *= IP_factor
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_IPe[ii] = np.array(self.in_IPe[ii]) * IP_factor
                        else:
                            self.in_IPe[ii] = np.sum(self.in_IPe[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_IPe[ii] *= IP_factor
            if active['IPi']:
                IP_factor = self.charge_by_name[species[1]] / self.dz
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_IPi = np.array(self.in_IPi[1]) * IP_factor
                    else:
                        collections = len(self.in_IPi)
                        self.in_IPi = np.sum(self.in_IPi, axis=0) / collections
                        self.in_IPi *= IP_factor
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_IPi[ii] = np.array(self.in_IPi[ii]) * IP_factor
                        else:
                            self.in_IPi[ii] = np.sum(self.in_IPi[ii], axis=0) / self.num_in_collections_this_output[ii]
                            self.in_IPi[ii] *= IP_factor
            if active['J_d']:
                if len(self.in_slices) == 1:
                    if self.num_in_collections_this_output[0] == 0:
                        pass
                    elif self.num_in_collections_this_output[0] == 1:
                        self.in_J_d = np.array(self.in_J_d[1]) * (constants.ep0 / self.dt)
                    else:
                        collections = len(self.in_J_d)
                        self.in_J_d = np.sum(self.in_J_d, axis=0) / collections * (constants.ep0 / self.dt)
                else:
                    for ii in range(len(self.in_slices)):
                        if self.num_in_collections_this_output[ii] == 0:
                            pass
                        elif self.num_in_collections_this_output[ii] == 1:
                            self.in_J_d[ii] = np.array(self.in_J_d[ii]) * (constants.ep0 / self.dt)
                        else:
                            self.in_J_d[ii] = np.sum(self.in_J_d[ii], axis=0) / self.num_in_collections_this_output[ii] * (constants.ep0 / self.dt)

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
        if self.Riz_switch:
            for species in self.species_names[1:]:
                np.save(os.path.join(self.Riz_dir_by_species[species], f'Riz_{step:04d}.npy'), self.Riz_by_species[species])

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
        outs = self.num_in_collections_this_output
        for key in active:
            if active[key]:
                arrays_dict = {f't{i+1:02d}': getattr(self, f'in_{key}')[i] for i in range(len(self.in_slices)) if outs[i] > 0}
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