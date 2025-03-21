#!/usr/bin/env python3
import argparse

import numpy as np
import sys, os, time, copy

from pywarpx import callbacks, fields, libwarpx, particle_containers, picmi
from mpi4py import MPI as mpi

from picmi_diagnostics.main import Diagnostics1D

comm = mpi.COMM_WORLD
num_proc = comm.Get_size()

constants = picmi.constants

milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12

eV_in_K = 11605.41586
ng_1Torr = 322.3e20                     # 1 Torr in m^-3

class CapacitiveDischargeExample(object):

    zmin            = 0.0                               # m
    zmax            = 67*milli                          # m
    freq            = 13.56e6                           # Hz
    voltage_rf      = 450.0                             # V
    gas_density     = 30.0*ng_1Torr*milli               # [mTorr]
    gas_temp        = 300.0                             # [K]
    m_ion           = 6.67e-27                          # [kg]

    plasma_density  = 2.56e14                           # [m^-3]
    elec_temp       = 2.585 * eV_in_K                   # [eV] to [K]

    seed_nppc       = 512                               # Number of particles per cell

    lambda_De       = np.sqrt(constants.ep0 * constants.kb * 2 * elec_temp / (2 * plasma_density * constants.q_e**2))
    omega_p         = np.sqrt(2 * plasma_density * constants.q_e**2 / (constants.ep0 * constants.m_e))

    dz              = zmax/128                           # Cell size
    nz              = int(zmax / dz)                     # Number of cells

    dt              = 1/(400*freq)                      # [s]

    convergence_time = 1280 / freq                      # Convergence time
    evolve_time = 0.0                                   # Time to evolve between diagnostic evaluations
    diag_time = 32 / freq                               # Time of diagnostic evaluations

    num_diag_steps = 1                                  # Number of diagnostic evaluations
    collections_per_diag_step = 3200                    # Number of collections per diagnostic evaluation for time resolved diagnostics
    interval_diag_times = [0, 0.25, 0.5, 0.75]          # Times to evaluate interval diagnostics (as a fraction of the RF period), if turned on
    interval_time = 1 / freq                            # Time to run interval diagnostics
    Riz_collection_time = diag_time                     # Time to run ionization rate diagnostics

    # Wall distribution function collection parameters
    ieadf_max_eV     = 40                                # Maximum energy for ion energy distribution function [eV]
    num_bins_ieadf   = 120                               # Number of bins for ion energy distribution function

    # Normal distribution function collection parameters
    eedf_max_eV     = 40                                # Maximum energy for electron energy distribution function [eV]
    iedf_max_eV     = 40                                # Maximum energy for ion energy distribution function [eV]
    num_bins        = 120                               # Number of bins for both distributions
    edf_boundaries  = []                                # Boundaries for spatially resolved edfs (if empty, will make one for the full domain)

    restart_checkpoint = False                          # Restart from checkpoint
    path_to_checkpoint = 'checkpoints/chkpt00000000'    # Path to desired checkpoint directory ending with the step number

    # Total simulation time in seconds
    total_time = convergence_time + (num_diag_steps - 1) * (diag_time + evolve_time) + diag_time

    bonus_steps        = 4                              # Number of extra steps to run after the last diagnostic ends (to ensure everything is saved)

    # Set switches for custom diagnostics
    # Create switches for custom diagnostics
    diag_switches = {
        'ieadfs': {
            'z_lo': False,
            'z_hi': False,
        },
        'rate_ioniz' : True,
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
            'J_w': False,
            'CPe': False,
            'CPi': False,
            'IPe': False,
            'IPi': False,
            'EEdf': False,
            'IEdf': False
        },
        'time_resolved': {
            'N_i': True,
            'N_e': False,
            'E_z': False,
            'phi': True,
            'W_e': True,
            'W_i': True,
            'Jze': True,
            'Jzi': True,
            'J_d': True,
            'J_w': False,
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
            'J_w': False,
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

    def __init__(self, verbose=False):
        '''Get input parameters for the specific case (n) desired.'''
        # Control verbose (-v) flag output
        self.verbose = verbose

        # Case specific input parameters
        self.voltage = f'{self.voltage_rf}*sin(2*pi*{self.freq:.5e}*t)'

        # Set MCC subcycling steps
        self.mcc_subcycling_steps = None

        # Get the starting step number
        self.start_step = 0
        if self.restart_checkpoint:
            self.start_step += int(self.path_to_checkpoint.strip('/')[-8:])

        # Get the (approximate) maximum step (will be updated later)
        self.max_steps = int(self.total_time / self.dt) + self.start_step + self.bonus_steps

        self.setup_run()

    def _setup_diagnostic_steps(self):
        '''
        Get step numbers to perform diagnostics. Computes:
        - self.diag_start: first step of diagnostic collection for each
          output set
        - self.diag_stop: last step of diagnostic collection for each
          output set (and the step info is saved)
        - self.diag_period_steps: number of steps between diagnostics
        '''
        # Make sure we run at least one convergence step
        if int(self.convergence_time/self.dt) == 0:
            self.convergence_time = self.dt

        # Setup local variables
        diag_start = self.convergence_time
        diag_n_evolve = self.diag_time + self.evolve_time

        # Account for the restart checkpoint
        if self.restart_checkpoint:
            diag_start += self.sim.extension.warpx.gett_new(lev=0)
            self.total_time += self.sim.extension.warpx.gett_new(lev=0)
            self.max_steps = int(self.total_time / self.dt) + self.bonus_steps

        # Calculate timesteps
        self.convergence_steps = int(diag_start / self.dt)
        self.evolve_steps = int(self.evolve_time / self.dt)

        # Make a list of diagnostic start times
        diag_start_times = []
        for i in range(self.num_diag_steps):
            diag_start_times.append(diag_start + i * diag_n_evolve)
        diag_start_times = np.array(diag_start_times)

        # Convert times to steps
        self.diag_start = np.round(diag_start_times / self.dt).astype(int)
        self.diag_period_steps = int(self.diag_time / self.dt)
        self.diag_stop = self.diag_start + self.diag_period_steps

        # If diag_stop[ii] == diag_start[ii+1], shift diag_end[ii] back 1 step
        for ii in range(1, len(self.diag_start)):
            if self.diag_start[ii] == self.diag_stop[ii - 1]:
                self.diag_stop[ii - 1] -= 1

        # If we are restarting and not doing any convergence, move the first diagnostic up one step
        if self.restart_checkpoint and self.diag_start[0] < self.start_step:
            self.diag_start[0] = self.start_step

    def setup_run(self):
        '''Setup simulation components.'''

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian1DGrid(
            number_of_cells=[self.nz],
            # warpx_blocking_factor=self.blocking_factor,
            lower_bound=[self.zmin],
            upper_bound=[self.zmax],
            lower_boundary_conditions=['dirichlet'],
            upper_boundary_conditions=['dirichlet'],
            lower_boundary_conditions_particles=['absorbing'],
            upper_boundary_conditions_particles=['absorbing'],
            warpx_potential_lo_z=0.0,
            warpx_potential_hi_z=self.voltage,
        )

        #######################################################################
        # Field setup                                                         #
        #######################################################################

        # This will use the tridiagonal solver
        self.solver = picmi.ElectrostaticSolver(grid=self.grid)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        elec_distribution = picmi.UniformDistribution(
            density=self.plasma_density,
            rms_velocity=[np.sqrt(constants.kb * self.elec_temp / constants.m_e)]*3,
        )
        ion_distribution = picmi.UniformDistribution(
            density=self.plasma_density,
            rms_velocity=[np.sqrt(constants.kb * self.gas_temp / self.m_ion)]*3,
        )

        self.electrons = picmi.Species(
            particle_type='electron', name='electrons',
            initial_distribution=elec_distribution
        )
        self.ions = picmi.Species(
            particle_type='He', name='he_ions',
            charge='q_e', mass=self.m_ion,
            initial_distribution=ion_distribution,
            warpx_save_particles_at_zhi = True,
            warpx_save_particles_at_zlo = True,
            warpx_add_real_attributes = {'orig_z': f'z',
                                         'orig_t': f't'}
        )

        #######################################################################
        # Collision initialization                                            #
        #######################################################################

        cross_sec_direc = '/home/bj8080/src/warpx-data/MCC_cross_sections/He/'
        electron_colls = picmi.MCCCollisions(
            name='coll_elec',
            species=self.electrons,
            background_density=self.gas_density,
            background_temperature=self.gas_temp,
            background_mass=self.ions.mass,
            ndt=self.mcc_subcycling_steps,
            scattering_processes={
                'elastic' : {
                    'cross_section' : cross_sec_direc+'electron_scattering.dat'
                },
                'excitation1' : {
                    'cross_section': cross_sec_direc+'excitation_1.dat',
                    'energy' : 19.82
                },
                'excitation2' : {
                    'cross_section': cross_sec_direc+'excitation_2.dat',
                    'energy' : 20.61
                },
                'ionization' : {
                    'cross_section' : cross_sec_direc+'ionization.dat',
                    'energy' : 24.55,
                    'species' : self.ions
                },
            }
        )

        ion_scattering_processes={
            'elastic': {'cross_section': cross_sec_direc+'ion_scattering.dat'},
            'back': {'cross_section': cross_sec_direc+'ion_back_scatter.dat'},
        }
        ion_colls = picmi.MCCCollisions(
            name='coll_ion',
            species=self.ions,
            background_density=self.gas_density,
            background_temperature=self.gas_temp,
            ndt=self.mcc_subcycling_steps,
            scattering_processes=ion_scattering_processes
        )

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        if self.restart_checkpoint:
            self.sim = picmi.Simulation(
                solver=self.solver,
                time_step_size=self.dt,
                max_steps=self.max_steps,
                warpx_collisions=[electron_colls, ion_colls],
                verbose=self.verbose,
                warpx_break_signals = 'USR1',
                warpx_numprocs = [num_proc],
                warpx_field_gathering_algo = 'energy-conserving',
                warpx_amr_restart = self.path_to_checkpoint
            )
        else:
            self.sim = picmi.Simulation(
                solver=self.solver,
                time_step_size=self.dt,
                max_steps=self.max_steps,
                warpx_collisions=[electron_colls, ion_colls],
                verbose=self.verbose,
                warpx_break_signals = 'USR1',
                warpx_numprocs = [num_proc],
                warpx_field_gathering_algo = 'energy-conserving'
            )
        self.solver.sim = self.sim

        self.sim.add_species(
            self.electrons,
            layout = picmi.GriddedLayout(
                n_macroparticle_per_cell=[self.seed_nppc], grid=self.grid
            )
        )
        self.sim.add_species(
            self.ions,
            layout = picmi.GriddedLayout(
                n_macroparticle_per_cell=[self.seed_nppc], grid=self.grid
            )
        )
        self.solver.sim_ext = self.sim.extension

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################
        const_diag = picmi.FieldDiagnostic(
            name = 'periodic',
            grid = self.grid,
            period = f'{self.start_step}::{(self.max_steps - self.start_step) // 40}',
            data_list = ['phi','rho_he_ions'],
            write_dir = './diags',
            warpx_format = 'openpmd',
            warpx_file_min_digits = 8
        )
        self.sim.add_diagnostic(const_diag)

        checkpoint = picmi.Checkpoint(
            name = 'checkpt',
            period = f'{self.start_step}::{(self.max_steps - self.start_step) // 4}',
            write_dir = './checkpoints',
            warpx_file_min_digits = 8,
            warpx_file_prefix = f'chkpt'
        )
        # self.sim.add_diagnostic(checkpoint)

        # Initialize everything
        self.sim.initialize_inputs()
        self.sim.initialize_warpx()

        # Add timings for custom diagnostics
        self._setup_diagnostic_steps()

        # Add custom diagnostics
        self.picmi_diagnostics = Diagnostics1D(
            self,
            self.sim.extension,
            switches=self.diag_switches,
            interval_times=self.interval_diag_times,
            ion_spec_names=['he_ions'],
            restart_checkpoint=self.restart_checkpoint
        )

    #######################################################################
    # Run Simulation                                                      #
    #######################################################################
    def run_sim(self):

        elapsed_steps = 0
        if self.restart_checkpoint:
            elapsed_steps = self.sim.extension.warpx.getistep(lev=0)

        ### Run until convergence ###
        #############################
        self.sim.step(self.convergence_steps - elapsed_steps - 1)
        elapsed_steps = self.convergence_steps - 1

        # Set up the particle buffer for diagnostic collection
        particle_buffer = particle_containers.ParticleBoundaryBufferWrapper()
        particle_buffer.clear_buffer()

        callbacks.installafterstep(self.picmi_diagnostics.do_diagnostics)

        # Run the simulation until the end
        self.sim.step(self.max_steps - elapsed_steps + self.bonus_steps)

        callbacks.uninstallcallback('afterstep', self.picmi_diagnostics.do_diagnostics)


##########################
### Execute Simulation ###
##########################
parser = argparse.ArgumentParser()
parser.add_argument(
    '-v', help='Verbose run, default = False', action='store_true'
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1]+left

run = CapacitiveDischargeExample(
    verbose=args.v
)
run.run_sim()
