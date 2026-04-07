from dataclasses import field

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import os

class Analysis:
    def __init__(self, directory: str = './diags', quiet_startup: bool = False):
        '''
        Initialize the Analysis object with the directory of the diagnostics data

        Parameters
        ----------
        directory : str
            The directory of the diagnostics data
        quiet_startup : bool, default=False
            Display the startup information
        '''
        self._initialize_basic_attributes()
        self._setup_directory(directory)
        self._load_basic_parameters()
        self._initialize_wall_eadf_data(quiet_startup)
        self._initialize_ionization_rate_data(quiet_startup)
        self._initialize_interval_data(quiet_startup)
        self._initialize_time_resolved_data(quiet_startup)
        self._initialize_time_averaged_data(quiet_startup)
        self._load_spatial_grids()
        self._initialize_edf_data(quiet_startup)

    def _initialize_basic_attributes(self):
        '''Initialize basic boolean flags and cell diagnostics list'''
        self.cell_diags = ['E_z', 'J_d', 'P_C', 'P_I']
        self.wall_eadf_bool = False
        self.Riz_bool = False
        self.in_bool = False
        self.tr_bool = False
        self.ta_bool = False
        self.species_names = []
        self.ylabel_dict = {
            'N': 'Density [m$^{-3}$]',
            'W': 'Average Energy [eV]',
            'Jz': 'Current Density [A/m$^2$]',
            'P_C': 'Capacitive Power [W/m$^3$]',
            'P_I': 'Inductive Power [W/m$^3$]',
            'EDF': 'EDF [a.u.]',
            'ExDF': 'x-EDF [a.u.]',
            'EyDF': 'y-EDF [a.u.]',
            'EzDF': 'z-EDF [a.u.]',
            'E_z': 'Electric Field [V/m]',
            'phi': 'Potential [V]',
            'J_d': 'Displacement Current Density [A/m$^2$]',
            'coll-rate': 'Collision Rate [m$^{-3}$s$^{-1}$]',
            'coll-energy': 'Collisional Power Loss [W/m$^3$]',
        }

    def _setup_directory(self, directory: str):
        '''Set up directory paths and get directory listing'''
        self.directory = os.path.abspath(directory)
        self.dir = os.listdir(directory)
        self.dir.sort()

    def _load_basic_parameters(self):
        '''Load timestep and cell size from diagnostic_times.dat'''
        with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
            for line in f:
                if line.startswith('Timestep [s]='):
                    self.dt = float(line.split('=')[1])
                    break
        with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
            for line in f:
                if line.startswith('Cell size [m]='):
                    self.dz = float(line.split('=')[1])
                    break
        with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
            for line in f:
                if line.startswith('Species:'):
                    self.species_names = [name.strip() for name in line.split(':')[1].split(',')]
                    break

    def _initialize_wall_eadf_data(self, quiet_startup: bool):
        '''Initialize Wall Energy Angular Distribution Function data'''
        if not any(d.startswith(('eadf', 'ieadf')) for d in self.dir):
            return

        if not quiet_startup:
            print('Wall EADF data found')
        self.wall_eadf_bool = True

        # Save the wall eadf directories (there will be one for each species)
        temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith(('eadf', 'ieadf'))]
        temp.sort()
        self.wall_eadf_dir = {}
        for species_dir in temp:
            # Save the species name as the dictionary key and the directory as the value
            self.wall_eadf_dir[species_dir.split('eadf_')[-1]] = species_dir

        if not quiet_startup:
            if len(self.wall_eadf_dir) > 1:
                print(f' - {len(self.wall_eadf_dir)} Wall EADF directories found for species: {", ".join(self.wall_eadf_dir.keys())}')
            else:
                print(f' - {len(self.wall_eadf_dir)} Wall EADF directory found for species: {", ".join(self.wall_eadf_dir.keys())}')

        # Initialize dictionaries
        self.wall_eadf_energy = {}
        self.wall_eadf_energy_edges = {}
        self.wall_eadf_deg = {}
        self.wall_eadf_deg_edges = {}
        self.lw_eadf_colls = {}
        self.rw_eadf_colls = {}
        self.wall_eadf_data_lists = {}

        # Process each species directory
        for key, directory in self.wall_eadf_dir.items():
            self._process_wall_eadf_species_directory(key, directory, quiet_startup)

    def _process_wall_eadf_species_directory(self, species: str, directory: str, quiet_startup: bool):
        '''Process Wall EADF data for a single species directory'''
        if not quiet_startup:
            print(f' - Looking into directory for species: {species}')

        wall_eadf_dir = os.listdir(directory)
        wall_eadf_dir.sort()

        # Load energy bins and create edges
        if 'bins_eV.npy' in wall_eadf_dir:
            self.wall_eadf_energy[species] = np.load(directory + '/bins_eV.npy')
            # Energies are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
            self.wall_eadf_energy_edges[species] = np.zeros(self.wall_eadf_energy[species].size + 1)
            self.wall_eadf_energy_edges[species][0] = self.wall_eadf_energy[species][0] - (self.wall_eadf_energy[species][1] - self.wall_eadf_energy[species][0])/2
            self.wall_eadf_energy_edges[species][1:-1] = (self.wall_eadf_energy[species][1:] + self.wall_eadf_energy[species][:-1])/2
            self.wall_eadf_energy_edges[species][-1] = self.wall_eadf_energy[species][-1] + (self.wall_eadf_energy[species][-1] - self.wall_eadf_energy[species][-2])/2
        elif not quiet_startup:
            print(f'   > Energy bins not found')

        # Load degree bins and create edges
        if 'bins_deg.npy' in wall_eadf_dir:
            self.wall_eadf_deg[species] = np.load(directory + '/bins_deg.npy')
            # Degrees are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
            self.wall_eadf_deg_edges[species] = np.zeros(self.wall_eadf_deg[species].size + 1)
            self.wall_eadf_deg_edges[species][0] = self.wall_eadf_deg[species][0] - (self.wall_eadf_deg[species][1] - self.wall_eadf_deg[species][0])/2
            self.wall_eadf_deg_edges[species][1:-1] = (self.wall_eadf_deg[species][1:] + self.wall_eadf_deg[species][:-1])/2
            self.wall_eadf_deg_edges[species][-1] = self.wall_eadf_deg[species][-1] + (self.wall_eadf_deg[species][-1] - self.wall_eadf_deg[species][-2])/2
        elif not quiet_startup:
            print(f'   > Degree bins not found')

        self.wall_eadf_data_lists[species] = {}

        # Process left wall collections
        if any(file.startswith('lw') for file in wall_eadf_dir):
            self.lw_eadf_colls[species] = [f'{directory}/{file}' for file in wall_eadf_dir if file.startswith('lw')]
            self.lw_eadf_colls[species].sort()
            if not quiet_startup:
                print(f'   > {len(self.lw_eadf_colls[species])} left wall collections')
            self.wall_eadf_data_lists[species]['lw'] = []

        # Process right wall collections
        if any(file.startswith('rw') for file in wall_eadf_dir):
            self.rw_eadf_colls[species] = [f'{directory}/{file}' for file in wall_eadf_dir if file.startswith('rw')]
            self.rw_eadf_colls[species].sort()
            if not quiet_startup:
                print(f'   > {len(self.rw_eadf_colls[species])} right wall collections')
            self.wall_eadf_data_lists[species]['rw'] = []

    def _initialize_ionization_rate_data(self, quiet_startup: bool):
        '''Initialize ionization rate data'''
        if not any(dir.startswith('r_ioniz') for dir in self.dir):
            return

        if not quiet_startup:
            print('Ionization rate data found')
        self.Riz_bool = True

        # Save the r_ioniz directories (there will be one for each ion species)
        temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('r_ioniz')]
        temp.sort()
        self.Riz_dir = {}
        for species_dir in temp:
            # Save the species name as the dictionary key and the directory as the value
            self.Riz_dir[species_dir.split('r_ioniz_')[-1]] = species_dir

        if not quiet_startup:
            if len(self.Riz_dir) > 1:
                print(f' - {len(self.Riz_dir)} Ionization rate directories found for species: {", ".join(self.Riz_dir.keys())}')
            else:
                print(f' - {len(self.Riz_dir)} Ionization rate directory found for species: {", ".join(self.Riz_dir.keys())}')

        # Initialize dictionaries
        self.Riz_z = {}
        self.Riz_z_edges = {}
        self.Riz_t = {}
        self.Riz_t_edges = {}
        self.Riz_colls = {}
        self.Riz_data_lists = {}

        # Process each species directory
        for key, directory in self.Riz_dir.items():
            self._process_ionization_species_directory(key, directory, quiet_startup)

    def _process_ionization_species_directory(self, species: str, directory: str, quiet_startup: bool):
        '''Process ionization rate data for a single species directory'''
        if not quiet_startup:
            print(f' - Looking into directory for species: {species}')

        Riz_dir = os.listdir(directory)
        Riz_dir.sort()

        # Load position bins and create edges
        if 'bins_z.npy' in Riz_dir:
            self.Riz_z[species] = np.load(directory + '/bins_z.npy')
            # Positions are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
            self.Riz_z_edges[species] = np.zeros(self.Riz_z[species].size + 1)
            self.Riz_z_edges[species][0] = self.Riz_z[species][0] - (self.Riz_z[species][1] - self.Riz_z[species][0])/2
            self.Riz_z_edges[species][1:-1] = (self.Riz_z[species][1:] + self.Riz_z[species][:-1])/2
            self.Riz_z_edges[species][-1] = self.Riz_z[species][-1] + (self.Riz_z[species][-1] - self.Riz_z[species][-2])/2
        elif not quiet_startup:
            print(f'   > Position bins not found')

        # Load time bins and create edges
        if 'bins_t.npy' in Riz_dir:
            self.Riz_t[species] = np.load(directory + '/bins_t.npy')
            # Times are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
            self.Riz_t_edges[species] = np.zeros(self.Riz_t[species].size + 1)
            self.Riz_t_edges[species][0] = self.Riz_t[species][0] - (self.Riz_t[species][1] - self.Riz_t[species][0])/2
            self.Riz_t_edges[species][1:-1] = (self.Riz_t[species][1:] + self.Riz_t[species][:-1])/2
            self.Riz_t_edges[species][-1] = self.Riz_t[species][-1] + (self.Riz_t[species][-1] - self.Riz_t[species][-2])/2
        elif not quiet_startup:
            print(f'   > Time bins not found')

        self.Riz_data_lists[species] = {}

        # Process data collections
        if any(file.startswith('Riz') for file in Riz_dir):
            self.Riz_colls[species] = [f'{directory}/{file}' for file in Riz_dir if file.startswith('Riz')]
            self.Riz_colls[species].sort()
            if not quiet_startup:
                print(f'   > {len(self.Riz_colls[species])} data collections')
            self.Riz_data_lists[species] = []

    def _initialize_interval_data(self, quiet_startup: bool):
        '''Initialize interval data'''
        if not any(dir.startswith('interval') for dir in self.dir):
            return

        if not quiet_startup:
            print('Interval data found')
        self.in_bool = True

        temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('interval')]
        temp.sort()
        self.in_colls = {}
        for coll in temp:
            self.in_colls[int(coll.split('/')[-1].split('_')[-1])] = coll

        num_colls = len(self.in_colls)
        if num_colls == 0:
            if not quiet_startup:
                print(f' - {num_colls} interval collections found')
            return

        # Load time intervals
        with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
            for line in f:
                if line.startswith('Times in interval='):
                    self.in_times = np.array([float(time) for time in line.split('=')[1].split(', ')])
                    break

        if not quiet_startup:
            print(f' - {num_colls} interval collections at {len(self.in_times)} time intervals: {", ".join([str(time) for time in self.in_times])}')

        # Get field names
        self.in_fields = [file.split('.')[0] for file in os.listdir(self.in_colls[1]) if file.endswith('.npz')]
        self.in_fields.sort()
        if not quiet_startup:
            print(f' - {len(self.in_fields)} fields: {", ".join(self.in_fields)}')

        # Set up dictionary to store interval data
        self.in_data = {}
        for field in self.in_fields:
            self.in_data[field] = {}
            for collection in self.in_colls:
                self.in_data[field][collection] = [0]*len(self.in_times)

    def _initialize_time_resolved_data(self, quiet_startup: bool):
        '''Initialize time resolved data'''
        if not any(dir.startswith('time_resolved') for dir in self.dir):
            return

        if not quiet_startup:
            print('Time resolved data found')
        self.tr_bool = True

        temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('time_resolved')]
        temp.sort()
        self.tr_colls = {}
        for coll in temp:
            self.tr_colls[int(coll.split('/')[-1].split('_')[-1])] = coll

        num_colls = len(self.tr_colls)
        if not quiet_startup:
            print(f' - {num_colls} time resolved collections')

        if num_colls == 0:
            return

        # Get field names
        self.tr_fields = [file.split('.')[0] for file in os.listdir(self.tr_colls[1]) if file.endswith('.npy') and file != 'times.npy']
        self.tr_fields.sort()
        if not quiet_startup:
            print(f' - {len(self.tr_fields)} fields: {", ".join(self.tr_fields)}')

        # Set up dictionaries
        self.tr_data = {}
        for field in self.tr_fields:
            self.tr_data[field] = {}
            for collection in self.tr_colls:
                self.tr_data[field][collection] = []

        # Load times for each collection
        self.tr_times = {}
        for collection in self.tr_data[field]:
            self.tr_times[collection] = np.load(f'{self.tr_colls[collection]}/times.npy')

        # Get the interval period
        with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
            for line in f:
                if line.startswith('Interval period [s]='):
                    self.interval_period = float(line.split('=')[1])
                    break
        if not quiet_startup:
            print(f' - Assuming an RF period of {self.interval_period:.2e} s')

    def _initialize_time_averaged_data(self, quiet_startup: bool):
        '''Initialize time averaged data'''
        if not any(dir.startswith('time_averaged') for dir in self.dir):
            return

        if not quiet_startup:
            print('Time averaged data found')
        self.ta_bool = True

        temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('time_averaged')]
        temp.sort()
        self.ta_colls = {}
        for coll in temp:
            self.ta_colls[int(coll.split('/')[-1].split('_')[-1])] = coll

        if not quiet_startup:
            print(f' - {len(self.ta_colls)} time averaged collections')

        # Get field names
        self.ta_fields = [file.split('.')[0] for file in os.listdir(self.ta_colls[1]) if file.endswith('.npy')]
        self.ta_fields.sort()
        if not quiet_startup:
            print(f' - {len(self.ta_fields)} fields: {", ".join(self.ta_fields)}')

        # Set up dictionary to store time averaged data
        self.ta_data = {}
        for field in self.ta_fields:
            self.ta_data[field] = {}
            for collection in self.ta_colls:
                self.ta_data[field][collection] = []

    def _load_spatial_grids(self):
        '''Load spatial grid data if any time-based data exists'''
        if self.in_bool or self.tr_bool or self.ta_bool:
            self.cells = np.load(f'{self.directory}/cells.npy')
            self.nodes = np.load(f'{self.directory}/nodes.npy')

    def _initialize_edf_data(self, quiet_startup: bool):
        '''Initialize energy distribution function data'''
        # Check if any EDF data exists
        has_edf_data = False

        for attr in ['ta_fields', 'tr_fields', 'in_fields']:
            if hasattr(self, attr):
                fields = getattr(self, attr)
                if any(field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')) for field in fields):
                    has_edf_data = True
                break

        if not has_edf_data:
            return

        if not quiet_startup:
            print('Energy distribution function data found')

        # Get the boundaries of the df collection region from the diagnostic_times.dat file
        self.edf_box_boundaries = []
        with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
            # First loop: find the line with the marker
            for line in f:
                if 'EDF Boundaries [m]:' in line:
                    # Remove 'EDF Boundaries [m]:' and extract data
                    data_part = line.split(':')[-1]
                    self.edf_box_boundaries.append(np.array(data_part.strip().strip('[]').split(), dtype=float))
                    break  # Exit this loop once the marker line is processed
            # Second loop: read subsequent lines until an empty line or EOF
            for line in f: # Continues from where the previous loop left off
                if line.strip() == '': # Check for an empty line
                    break # Stop if an empty line is found
                self.edf_box_boundaries.append(np.array(line.strip().strip('[]').split(), dtype=float))
            self.edf_box_boundaries = np.concatenate(self.edf_box_boundaries)

        # Process EDF boundaries and indices
        self.edf_boundary_node_indices = np.r_[0, np.searchsorted(self.nodes, self.edf_box_boundaries, side='left'), len(self.nodes)-1]
        self.edf_box_boundaries = np.concatenate(([0], self.edf_box_boundaries, [self.nodes[-1]]))
        self.num_edfs = len(self.edf_box_boundaries) - 1
        if not quiet_startup:
            print(f' - Edfs collected in {self.num_edfs} regions')

        # Calculate midpoints and indices
        self.edf_box_midpoints = (self.edf_box_boundaries[:-1] + self.edf_box_boundaries[1:]) / 2
        self.edf_midpoint_node_indices = np.searchsorted(self.nodes, self.edf_box_midpoints, side='left')

        # Load energy bins for relevant EDF types
        self.edf_energy = {}
        edf_field_start = ['EDF_' + species for species in self.species_names]
        for ii, edf_type in enumerate(edf_field_start):
            # Check if this EDF type exists in any of the field collections
            edf_exists = False
            for attr in ['ta_fields', 'tr_fields', 'in_fields']:
                if hasattr(self, attr):
                    fields = getattr(self, attr)
                    if any(field.startswith(edf_type) for field in fields):
                        edf_exists = True
                        break

            if edf_exists:
                try:
                    self.edf_energy[edf_type] = np.load(f'{self.directory}/edf_bins_eV_{self.species_names[ii]}.npy')
                except FileNotFoundError:
                    if ii == 0:
                        self.edf_energy[edf_type] = np.load(f'{self.directory}/eedf_bins_eV.npy')
                    else:
                        self.edf_energy[edf_type] = np.load(f'{self.directory}/iedf_bins_eV.npy')
            if not quiet_startup:
                print(f' - {edf_type} energy bins collected')

        # Load energy bins for relevant EDF types
        evdf_field_start = [f'{comp}DF_{species}' for species in self.species_names for comp in ['Ex', 'Ey', 'Ez']]
        for ii, evdf_type in enumerate(evdf_field_start):
            # Check if this EDF type exists in any of the field collections
            evdf_exists = False
            for attr in ['ta_fields', 'tr_fields', 'in_fields']:
                if hasattr(self, attr):
                    fields = getattr(self, attr)
                    if any(field.startswith(evdf_type) for field in fields):
                        evdf_exists = True
                        break

            if evdf_exists:
                self.edf_energy[evdf_type] = np.load(f'{self.directory}/{evdf_type.split("DF")[0].lower()}df_bins_eV_{evdf_type.split("_")[-1]}.npy')
            if not quiet_startup:
                print(f' - {evdf_type} energy bins collected')

    def load_Riz_data_lists(self, species: str = None):
        '''
        Load the ionization rate data

        Parameters
        ----------
        species : str, default=None
            The species to load

        Returns
        -------
        Riz_data_lists : dict[dict[list[np.ndarray]]]
            The ionization rate data organized like
            Riz_data_lists[species][wall][collection]
        '''
        if not self.Riz_bool:
            raise ValueError('Ionization rate data not found')
        if species is not None:
            if species not in self.Riz_dir:
                raise ValueError(f'Species must be one of: {", ".join(self.Riz_dir.keys())}')
            # Add data
            self.Riz_data_lists[species] = [np.load(coll) for coll in self.Riz_colls[species]]
        else:
            for spec in self.Riz_dir:
                self.Riz_data_lists[spec] = [np.load(coll) for coll in self.Riz_colls[spec]]

        return self.Riz_data_lists

    def get_avg_Riz_data(self):
        '''
        Get the average ionization rate data over all collections.

        Returns
        -------
        avg_Riz_data : dict[np.ndarray]
            The averaged ionization rate data for each species.
        '''
        if not self.Riz_bool:
            raise ValueError('Ionization rate data not found')
        # Load the Riz data
        self.load_Riz_data_lists()

        # Initialize the dictionary to store the average data
        self.avg_Riz_data = {}
        for species in self.Riz_data_lists:
            temp_array_list = []
            for array in self.Riz_data_lists[species]:
                temp_array_list.append(array)
            self.avg_Riz_data[species] = np.mean(temp_array_list, axis=0)

        return self.avg_Riz_data

    def get_Riz_vs_z_data_lists(self):
        '''
        Gets ionization rate versus z data from the full ionization rate
        data.

        Returns
        -------
        Riz_vs_z_data_lists : dict[list[np.ndarray]]
            The ionization rate data integrated over time
        '''
        if not self.Riz_bool:
            raise ValueError('Ionization rate data not found')
        self.load_Riz_data_lists()

        self.Riz_vs_z_data_lists = {}
        for species in self.Riz_data_lists:
            self.Riz_vs_z_data_lists[species] = []
            for array in self.Riz_data_lists[species]:
                self.Riz_vs_z_data_lists[species].append(np.sum(array, axis=0))
        return self.Riz_vs_z_data_lists

    def get_avg_Riz_vs_z_data(self):
        '''
        Get the average ionization rate versus z data over all collections.

        Returns
        -------
        avg_Riz_vs_z_data : dict[np.ndarray]
            The Riz vs position data for each species.
        '''
        if not self.Riz_bool:
            raise ValueError('Ionization rate data not found')
        if not hasattr(self, 'Riz_vs_z_data_lists'):
            self.get_Riz_vs_z_data_lists()

        # Initialize the dictionary to store the average data
        self.avg_Riz_vs_z_data = {}
        for species in self.Riz_vs_z_data_lists:
            self.avg_Riz_vs_z_data[species] = {}
            for wall in self.Riz_vs_z_data_lists[species]:
                temp_array_list = []
                for array in self.Riz_vs_z_data_lists[species]:
                    temp_array_list.append(array)
                self.avg_Riz_vs_z_data[species] = np.mean(temp_array_list, axis=0)
        return self.avg_Riz_vs_z_data

    def plot_avg_Riz_vs_z(self,
                          species: str = None,
                          dpi=150):
        '''
        Plot the collection-averaged ionization rate vs position data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on a single axis
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if not self.Riz_bool:
            raise ValueError('Ionization rate data not found')
        if not hasattr(self, 'avg_Riz_vs_z_data'):
            self.get_avg_Riz_vs_z_data()
        if species is not None and species not in self.avg_Riz_vs_z_data:
            raise ValueError(f'Species must be one of: {", ".join(self.avg_Riz_vs_z_data.keys())}')

        Riz = self.avg_Riz_vs_z_data
        if species is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            for spec in Riz:
                ax.plot(self.Riz_z[spec], Riz[spec], label = spec)
                ax.set_ylim(0, np.max(Riz[spec])*1.05)
            ax.set_xlabel('Position [m]')
            ax.set_ylabel('$R_i$ [m$^{-3}$s$^{-1}$]')
            ax.set_title('Ionization Rate')
            ax.legend()
            ax.margins(x=0)
        else:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            ax.plot(self.wall_eadf_energy[species], Riz[species], label = species)
            ax.set_ylim(0, np.max(Riz[species])*1.05)
            ax.set_xlabel('Position [m]')
            ax.set_ylabel('$R_i$ [m$^{-3}$s$^{-1}$]')
            ax.set_title('Ionization Rate')
            ax.legend()
            ax.margins(x=0)
        return fig, ax

    def plot_avg_Riz(self,
                     species: str = None,
                     dpi=150):
        '''
        Plot the collection-averaged ionization rate data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on separate figs
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure or list[matplotlib.figure.Figure]
            The figure object
        ax : matplotlib.axes.Axes or list[matplotlib.axes.Axes]
            The axes object
        '''
        if not self.Riz_bool:
            raise ValueError('Ionization rate data not found')
        if not hasattr(self, 'avg_Riz_data'):
            self.get_avg_Riz_data()
        if species is not None and species not in self.avg_Riz_data:
            raise ValueError(f'Species must be one of: {", ".join(self.avg_Riz_data.keys())}')

        else:
            Riz = self.avg_Riz_data
        if species is None:
            figs = []
            axs = []
            for spec in Riz:
                fig, ax = plt.subplots(1,1, dpi=dpi)
                figs.append(fig)
                axs.append(ax)
                cbar = ax.pcolormesh(self.Riz_z_edges[spec], self.Riz_t_edges[spec], Riz[spec], shading='auto')
                fig.colorbar(cbar, ax=ax, label='$R_i$ [m$^{-3}$s$^{-1}$]')
                ax.set_xlabel('Position [m]')
                ax.set_ylabel(r'Time in RF Period [t/$\tau_{RF}$]')
                ax.set_title(f'{spec} Ionization Rate')
            return figs, axs
        else:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            cbar = ax.pcolormesh(self.Riz_z_edges[species], self.Riz_t_edges[species], Riz[species], shading='auto')
            fig.colorbar(cbar, ax=ax, label='$R_i$ [m$^{-3}$s$^{-1}$]')
            ax.set_xlabel('Position [m]')
            ax.set_ylabel(r'Time in RF Period [t/$\tau_{RF}$]')
            ax.set_title(f'{species} Ionization Rate')
            return fig, ax

    def load_wall_eadf_data_lists(self, species: str = None):
        '''
        Load the wall EADF data

        Parameters
        ----------
        species : str, default=None
            The species to load

        Returns
        -------
        wall_eadf_data_lists : dict[dict[list[np.ndarray]]]
            The wall EADF data organized like
            wall_eadf_data_lists[species][wall][collection]
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        if species is not None:
            if species not in self.wall_eadf_dir:
                raise ValueError(f'Species must be one of: {", ".join(self.wall_eadf_dir.keys())}')
            # Add left wall data, if necessary
            if 'lw' in self.wall_eadf_data_lists[species]:
                self.wall_eadf_data_lists[species]['lw'] = [np.load(coll) for coll in self.lw_eadf_colls[species]]
            # Add right wall data, if necessary
            if 'rw' in self.wall_eadf_data_lists[species]:
                self.wall_eadf_data_lists[species]['rw'] = [np.load(coll) for coll in self.rw_eadf_colls[species]]
        else:
            for spec in self.wall_eadf_dir:
                if 'lw' in self.wall_eadf_data_lists[spec]:
                    self.wall_eadf_data_lists[spec]['lw'] = [np.load(coll) for coll in self.lw_eadf_colls[spec]]
                if 'rw' in self.wall_eadf_data_lists[spec]:
                    self.wall_eadf_data_lists[spec]['rw'] = [np.load(coll) for coll in self.rw_eadf_colls[spec]]

        return self.wall_eadf_data_lists

    def get_avg_wall_eadf_data(self, separate_rl: bool = False):
        '''
        Get the average wall EADF over all collections. Optionally, average the
        left and right wall data separately.

        Parameters
        ----------
        separate_rl : bool, default=False
            Average the left and right wall EADF data separately if True

        Returns
        -------
        avg_wall_eadf_data : dict[np.ndarray] or dict[dict[np.ndarray]]
            The averaged wall EADF data for each species. If separate_rl is False,
            the data is organized like avg_wall_eadf_data[species]. If separate_rl
            is True, the data is organized like avg_wall_eadf_data[species][wall].
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        # Load the wall eadf data
        self.load_wall_eadf_data_lists()

        # Average both walls data together
        if not separate_rl:
            # Initialize the dictionary to store the average wall EADF data
            self.avg_wall_eadf_data = {}
            for species in self.wall_eadf_data_lists:
                temp_array_list = []
                for wall in self.wall_eadf_data_lists[species]:
                    for array in self.wall_eadf_data_lists[species][wall]:
                        temp_array_list.append(array)
                # Save the average wall EADF data for the species
                self.avg_wall_eadf_data[species] = np.mean(temp_array_list, axis=0)
        else:
            # Initialize the dictionary to store the average wall EADF data
            self.avg_wall_eadf_data = {}
            for species in self.wall_eadf_data_lists:
                self.avg_wall_eadf_data[species] = {}
                for wall in self.wall_eadf_data_lists[species]:
                    temp_array_list = []
                    for array in self.wall_eadf_data_lists[species][wall]:
                        temp_array_list.append(array)
                    self.avg_wall_eadf_data[species][wall] = np.mean(temp_array_list, axis=0)

        return self.avg_wall_eadf_data

    def get_wall_edf_data_lists(self):
        '''
        Gets wall EDF data from the list of wall EDF data.

        Returns
        -------
        iedf_data_lists : dict[dict[list[np.ndarray]]]
            The IEDF data
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        self.load_wall_eadf_data_lists()
        self.wall_edf_data_lists = {}
        for species in self.wall_eadf_data_lists:
            self.wall_edf_data_lists[species] = {}
            for wall in self.wall_eadf_data_lists[species]:
                self.wall_edf_data_lists[species][wall] = []
                for array in self.wall_eadf_data_lists[species][wall]:
                    self.wall_edf_data_lists[species][wall].append(np.sum(array, axis=1))
        return self.wall_edf_data_lists

    def get_avg_wall_edf_data(self, separate_rl: bool = False):
        '''
        Get the average wall EDF over all collections. Optionally, average the
        left and right wall data separately.

        Parameters
        ----------
        separate_rl : bool, default=False
            Average the left and right wall EDF data separately if True

        Returns
        -------
        avg_wall_edf_data : dict[np.ndarray] or dict[dict[np.ndarray]]
            The wall EDF data for each species. If separate_rl is False, the data
            is organized like avg_wall_edf_data[species]. If separate_rl is True,
            the data is organized like avg_wall_edf_data[species][wall].
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EDF data not found')
        if not hasattr(self, 'iedf_data_lists'):
            self.get_wall_edf_data_lists()

        # Average both walls data together
        if not separate_rl:
            # Initialize the dictionary to store the average wall EDF data
            self.avg_wall_edf_data = {}
            for species in self.wall_edf_data_lists:
                temp_array_list = []
                for wall in self.wall_edf_data_lists[species]:
                    for array in self.wall_edf_data_lists[species][wall]:
                        temp_array_list.append(array)
                # Save the average IEDF data for the species
                self.avg_wall_edf_data[species] = np.mean(temp_array_list, axis=0)
        else:
            # Initialize the dictionary to store the average wall EDF data
            self.avg_wall_edf_data = {}
            for species in self.wall_edf_data_lists:
                self.avg_wall_edf_data[species] = {}
                for wall in self.wall_edf_data_lists[species]:
                    temp_array_list = []
                    for array in self.wall_edf_data_lists[species][wall]:
                        temp_array_list.append(array)
                    self.avg_wall_edf_data[species][wall] = np.mean(temp_array_list, axis=0)

        return self.avg_wall_edf_data

    def plot_avg_wall_edf(self,
                      species: str = None,
                      separate_rl: bool = False,
                      normalize: bool = True,
                      ax = None,
                      dpi=150):
        '''
        Plot the collection-averaged wall EDF data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on a single axis
        separate_rl : bool, default=False
            Average the left and right wall EDF data separately if True
        normalize : bool
            Normalize the wall EDF data
        ax: matplotlib.axes.Axes, default=None
            The axes to plot on. If None, creates a new figure and axes.
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        if not hasattr(self, 'avg_wall_edf_data'):
            self.get_avg_wall_edf_data(separate_rl=separate_rl)
        if species is not None and species not in self.wall_edf_data_lists:
            raise ValueError(f'Species must be one of: {", ".join(self.wall_edf_data_lists.keys())}')
        if normalize:
            edfs = self.normalize_wall_edf()
        else:
            edfs = self.avg_wall_edf_data

        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            return_fig = True

        if species is None:
            for spec in edfs:
                if isinstance(edfs[spec], dict):
                    for wall in edfs[spec]:
                        ax.plot(self.wall_eadf_energy[spec], edfs[spec][wall], label = f'{wall} {spec}')
                    ax.set_ylim(0, np.max([np.max(edfs[spec][wall]) for wall in edfs[spec]])*1.05)
                    ax.legend()
                else:
                    ax.plot(self.wall_eadf_energy[spec], edfs[spec], label = spec)
                    ax.set_ylim(0, np.max(edfs[spec])*1.05)
            ax.set_xlabel('Energy [eV]')
            ax.set_ylabel('EDF [eV$^{-1}$]')
            ax.set_title('Simulation Wall EDF')
            ax.margins(x=0)
        else:
            if isinstance(edfs[species], dict):
                for wall in edfs[species]:
                    ax.plot(self.wall_eadf_energy[species], edfs[species][wall], label = f'{wall} {species}')
                ax.set_ylim(0, np.max([np.max(edfs[species][wall]) for wall in edfs[species]])*1.05)
                ax.legend()
            else:
                ax.plot(self.wall_eadf_energy[species], edfs[species], label = species)
                ax.set_ylim(0, np.max(edfs[species])*1.05)
            ax.set_xlabel('Energy [eV]')
            ax.set_ylabel('EDF [eV$^{-1}$]')
            ax.set_title('Simulation Wall EDF')
            ax.margins(x=0)

        if return_fig:
            return fig, ax
        else:
            return ax

    def normalize_wall_edf(self):
        '''
        Normalize the collection-averaged Wall EDF data

        Returns
        -------
        iedf : dict[np.ndarray] or dict[dict[np.ndarray]]
            The normalized IEDF data, organized like iedf[species] or
            iedf[species][wall] based on how the data is organized coming in
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        if not hasattr(self, 'avg_wall_edf_data'):
            self.get_avg_wall_edf_data()
        self.normalized_wall_edfs = {}
        for species in self.avg_wall_edf_data:
            # Check if the species have been separated into left and right wall data
            if isinstance(self.avg_wall_edf_data[species], dict):
                self.normalized_wall_edfs[species] = {}
                for wall in self.avg_wall_edf_data[species]:
                    integral = np.trapezoid(self.avg_wall_edf_data[species][wall], self.wall_eadf_energy[species])
                    if integral > 0:
                        self.normalized_wall_edfs[species][wall] = self.avg_wall_edf_data[species][wall] / integral
                    else:
                        self.normalized_wall_edfs[species][wall] = np.zeros_like(self.avg_wall_edf_data[species][wall])
            else:
                integral = np.trapezoid(self.avg_wall_edf_data[species], self.wall_eadf_energy[species])
                if integral > 0:
                    self.normalized_wall_edfs[species] = self.avg_wall_edf_data[species] / integral
                else:
                    self.normalized_wall_edfs[species] = np.zeros_like(self.avg_wall_edf_data[species])

        return self.normalized_wall_edfs

    def plot_avg_wall_eadf(self,
                           species: str = None,
                           normalize: bool = True,
                           dpi=150):
        '''
        Plot the collection-averaged wall EADF data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on a separate figs
        normalize : bool
            Normalize the wall EADF data
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        if not hasattr(self, 'avg_wall_eadf_data'):
            self.get_avg_wall_eadf_data()
        if species is not None and species not in self.avg_wall_eadf_data:
            raise ValueError(f'Species must be one of: {", ".join(self.avg_wall_eadf_data.keys())}')
        if normalize:
            wall_eadfs = self.normalize_wall_eadf()
        else:
            wall_eadfs = self.avg_wall_eadf_data
        if species is None:
            figs = []
            axs = []
            for spec in wall_eadfs:
                if isinstance(wall_eadfs[spec], dict):
                    raise NotImplementedError('Cannot plot wall EADFs with separate left and right wall data yet. Needs to be implemented.')
                fig, ax = plt.subplots(1,1, dpi=dpi)
                figs.append(fig)
                axs.append(ax)
                cbar = ax.pcolormesh(self.wall_eadf_deg_edges[spec], self.wall_eadf_energy_edges[spec], wall_eadfs[spec], shading='auto')
                fig.colorbar(cbar, ax=ax, label='EADF [eV$^{-1}$]')
                ax.set_xlabel('Degrees')
                ax.set_ylabel('Energy [eV]')
                ax.set_title(f'{spec} EADF')
            return figs, axs
        else:
            if isinstance(wall_eadfs[species], dict):
                raise NotImplementedError('Cannot plot wall EADFs with separate left and right wall data yet. Needs to be implemented.')
            fig, ax = plt.subplots(1,1, dpi=dpi)
            cbar = ax.pcolormesh(self.wall_eadf_deg_edges[species], self.wall_eadf_energy_edges[species], wall_eadfs[species], shading='auto')
            fig.colorbar(cbar, ax=ax, label='EADF [eV$^{-1}$]')
            ax.set_xlabel('Degrees')
            ax.set_ylabel('Energy [eV]')
            ax.set_title(f'{species} EADF')
            return fig, ax

    def normalize_wall_eadf(self):
        '''
        Normalize the collection-averaged wall EADF data

        Returns
        -------
        wall_eadf : dict[np.ndarray]
            The normalized wall EADF data
        '''
        if not self.wall_eadf_bool:
            raise ValueError('Wall EADF data not found')
        if not hasattr(self, 'avg_wall_eadf_data'):
            self.get_avg_wall_eadf_data()
        self.normalized_wall_eadfs = {}
        for species in self.avg_wall_eadf_data:

            # Get the area factor to normalize the wall EADF data. To use, divide by the area factor.
            # Area factor is the sine of the angle multiplied by the square root of the energy
            area_factor = np.abs(np.sin(self.wall_eadf_deg[species] * np.pi / 180))
            area_factor = np.tile(area_factor, (self.wall_eadf_energy[species].size, 1)) # Resize area factor to be size (energy.size, deg.size)
            for ii in range(len(self.wall_eadf_energy[species])):
                area_factor[ii] = np.sqrt(self.wall_eadf_energy[species][ii]) * area_factor[ii] # Multiply each row by the corresponding energy bin to caluclate the area factor

            # Check if the species have been separated into left and right wall data
            if isinstance(self.avg_wall_eadf_data[species], dict):
                self.normalized_wall_eadfs[species] = {}
                for wall in self.avg_wall_eadf_data[species]:
                    self.normalized_wall_eadfs[species][wall] = self.avg_wall_eadf_data[species][wall] / np.trapz(np.trapz(self.avg_wall_eadf_data[species][wall], self.wall_eadf_energy[species], axis=0), self.wall_eadf_deg[species]) / area_factor
            else:
                self.normalized_wall_eadfs[species] = self.avg_wall_eadf_data[species] / np.trapz(np.trapz(self.avg_wall_eadf_data[species], self.wall_eadf_energy[species], axis=0), self.wall_eadf_deg[species]) / area_factor

        return self.normalized_wall_eadfs

    def load_intervals(self, field: str = None):
        '''
        Load the interval data

        Parameters
        ----------
        field : str
            The field to load, if None, loads all fields.

        Returns
        -------
        in_data : dict[dict[list[np.ndArray]]]
            The interval data
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')

        # Determine which fields to load
        if field is not None:
            if field not in self.in_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.in_fields)}')
            fields_to_load = [field]
        else:
            fields_to_load = self.in_fields

        # Load data for each requested field and collection
        for fld in fields_to_load:
            for coll in self.in_data[fld]:
                temp = np.load(f'{self.in_colls[coll]}/{fld}.npz')
                # Unpack elements of the npz file from t01 to t{self.interval_times.size+1} into list entries
                for ii in range(len(self.in_times)):
                    self.in_data[fld][coll][ii] = temp[f't{ii+1:02d}']
                    if fld.startswith('Jz') and fld != 'Jzc':
                        self.in_data[fld][coll][ii][0] *= 2
                        self.in_data[fld][coll][ii][-1] *= 2
        return self.in_data

    def add_interval_field(self, field: str):
        '''
        Add an interval field to the interval data

        Parameters
        ----------
        field : str
            The field to add. Must be one of 'P_t', 'EfV', 'Jzc', 'J_t'

        Returns
        -------
        in_data[field] : dict[stack of np.ndArray]
            The interval data
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        if field not in ['P_t', 'EfV', 'Jzc', 'J_t']:
            raise ValueError('Field must be one of: P_t, EfV, Jzc, J_t')
        if field == 'P_t':
            self.load_intervals('CPe')
            self.load_intervals('CPi')
            self.in_data[field] = {}
            for coll in self.in_data['CPe']:
                self.in_data[field][coll] = [0] * len(self.in_times)
                for interval in range(len(self.in_times)):
                    self.in_data[field][coll][interval] = np.sum((self.in_data['CPe'][coll][interval] + self.in_data['CPi'][coll][interval]) * self.dz)
            # Check if the field is already in self.in_fields before adding
            if field not in self.in_fields:
                self.in_fields.append(field)
        elif field == 'EfV':
            self.load_intervals('phi')
            self.in_data[field] = {}
            for coll in self.in_data['phi']:
                self.in_data[field][coll] = [0] * len(self.in_times)
                for interval in range(len(self.in_times)):
                    self.in_data[field][coll][interval] = -np.gradient(self.in_data['phi'][coll][interval], self.dz)
            # Check if the field is already in self.in_fields before adding
            if field not in self.in_fields:
                self.in_fields.append(field)
        elif field == 'Jzc':
            self.load_intervals('Jze')
            self.load_intervals('Jzi')
            self.in_data[field] = {}
            for coll in self.in_data['Jze']:
                self.in_data[field][coll] = [0] * len(self.in_times)
                for interval in range(len(self.in_times)):
                    self.in_data[field][coll][interval] = self.in_data['Jze'][coll][interval] + self.in_data['Jzi'][coll][interval]
            # Check if the field is already in self.in_fields before adding
            if field not in self.in_fields:
                self.in_fields.append(field)
        elif field == 'J_t':
            self.load_intervals('Jze')
            self.load_intervals('Jzi')
            self.load_intervals('J_d')
            self.in_data[field] = {}
            for coll in self.in_data['Jze']:
                self.in_data[field][coll] = [0] * len(self.in_times)
                for interval in range(len(self.in_times)):
                    # Interpolate J_d from cells to nodes
                    J_d_on_nodes = np.interp(self.nodes, self.cells, self.in_data['J_d'][coll][interval])
                    self.in_data[field][coll][interval] = self.in_data['Jze'][coll][interval] + self.in_data['Jzi'][coll][interval] + J_d_on_nodes
            # Check if the field is already in self.in_fields before adding
            if field not in self.in_fields:
                self.in_fields.append(field)

        return self.in_data[field]

    def avg_intervals(self, field: str = None):
        '''
        Average the interval data

        Parameters
        ----------
        field : str
            The field to average. Must be one of self.interval_fields

        Returns
        -------
        avg_in_data : dict
            The averaged interval data
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        if field is not None:
            # Check if field is a valid field
            if field not in self.in_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.in_fields)}')
            # Check if field has been loaded into self.interval_data
            if any([np.array_equal(self.in_data[field][coll][0], 0) for coll in self.in_data[field]]):
                self.load_intervals(field)
            # Check if self.avg_interval_data has been created yet
            if not hasattr(self, 'avg_in_data'):
                self.avg_in_data = {}
            # Make a dictionary entry for the current field
            self.avg_in_data[field] = []
            # For the field, go through at fixed time intervals and average the data. Append each average to self.avg_interval_data[fld]
            for ii in range(len(self.in_times)):
                self.avg_in_data[field].append(np.mean([self.in_data[field][coll][ii] for coll in self.in_data[field]], axis=0))
        else:
            # Set (or reset) self.avg_interval_data to an empty dictionary
            self.avg_in_data = {}
            for fld in self.in_fields:
                # Check if the current field has been loaded into self.interval_data
                if any([np.array_equal(self.in_data[fld][coll][0], 0) for coll in self.in_data[fld]]):
                    self.load_intervals(fld)
                # Make a dictionary entry for the current field
                self.avg_in_data[fld] = []
                # For the field, go through at fixed time intervals and average the data. Append each average to self.avg_interval_data[fld]
                for ii in range(len(self.in_times)):
                    self.avg_in_data[fld].append(np.mean([self.in_data[fld][coll][ii] for coll in self.in_data[fld]], axis=0))
        return self.avg_in_data

    def plot_avg_interval(self,
                          field: str,
                          interval: int = None,
                          plot_time_avg: bool = True,
                          ax = None,
                          dpi : int = 150,
                          cmap : str = 'GnBu'):
        '''
        Plot the average interval data

        Parameters
        ----------
        field : str
            The field to plot
        interval : int, default=None
            The index (from 0 to len(self.interval_times - 1)) of the interval
            to plot. If None, plots all intervals on a single axis.
        plot_time_avg : bool, default=True
            Plot the time-averaged data on the same axis
        ax : matplotlib.axes.Axes, default=None
            The axes object to plot on. If None, creates a new figure and axes
        dpi : int
            The DPI of the plot
        cmap : str, default='GnBu'
            The colormap to use, if plotting multiple intervals

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        return self.plot_phase_resolved(
            field=field, interval=interval, plot_time_avg=plot_time_avg,
            ax=ax, dpi=dpi, cmap=cmap)

    def load_time_resolved(self, field: str = None):
        '''
        Load the time resolved data

        Parameters
        ----------
        field : str
            The field to load, if None, loads all fields.

        Returns
        -------
        tr_data : dict[dict[stack of np.ndArray]]
            The time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')

        # Determine which fields to load
        if field is not None:
            if field not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            fields_to_load = [field]
        else:
            fields_to_load = self.tr_fields

        # Load data for each requested field and collection
        for fld in fields_to_load:
            for coll in self.tr_data[fld]:
                self.tr_data[fld][coll] = np.load(f'{self.tr_colls[coll]}/{fld}.npy')
                if fld.startswith('Jz') and fld != 'Jzc':
                    for ii in range(len(self.tr_data[fld][coll])):
                        self.tr_data[fld][coll][ii][0] *= 2
                        self.tr_data[fld][coll][ii][-1] *= 2
        return self.tr_data

    def add_time_resolved_field(self, field: str):
        '''
        Add a time resolved field to the time resolved data

        Parameters
        ----------
        field : str
            The field to add. Must be one of: 'P_t', 'EfV', 'Jzc', 'J_t'

        Returns
        -------
        tr_data : dict[dict[stack of np.ndArray]]
            The time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field not in ['P_t', 'EfV', 'Jzc', 'J_t']:
            raise ValueError('Field must be one of: P_t, EfV, Jzc, J_t')
        if field == 'P_t':
            self.load_time_resolved('CPe')
            self.load_time_resolved('CPi')
            self.tr_data[field] = {}
            for coll in self.tr_data['CPe']:
                self.tr_data[field][coll] = np.sum((self.tr_data['CPe'][coll] + self.tr_data['CPi'][coll]) * self.dz, axis=1)
            # Check if the field is already in self.tr_fields before adding
            if field not in self.tr_fields:
                self.tr_fields.append(field)
        elif field == 'EfV':
            self.load_time_resolved('phi')
            self.tr_data[field] = {}
            for coll in self.tr_data['phi']:
                self.tr_data[field][coll] = np.stack(-np.gradient(self.tr_data['phi'][coll], self.dt, self.dz, axis=(0,1))[1])
            # Check if the field is already in self.tr_fields before adding
            if field not in self.tr_fields:
                self.tr_fields.append(field)
        elif field == 'Jzc':
            self.load_time_resolved('Jze')
            self.load_time_resolved('Jzi')
            self.tr_data[field] = {}
            for coll in self.tr_data['Jze']:
                self.tr_data[field][coll] = self.tr_data['Jze'][coll] + self.tr_data['Jzi'][coll]
            # Check if the field is already in self.tr_fields before adding
            if field not in self.tr_fields:
                self.tr_fields.append(field)
        elif field == 'J_t':
            self.load_time_resolved('Jze')
            self.load_time_resolved('Jzi')
            self.load_time_resolved('J_d')
            self.tr_data[field] = {}
            for coll in self.tr_data['Jze']:
                # Interpolate J_d from cells to nodes for each time step
                J_d_on_nodes = np.array([np.interp(self.nodes, self.cells, J_d_timestep)
                                        for J_d_timestep in self.tr_data['J_d'][coll]])
                self.tr_data[field][coll] = self.tr_data['Jze'][coll] + self.tr_data['Jzi'][coll] + J_d_on_nodes
            # Check if the field is already in self.tr_fields before adding
            if field not in self.tr_fields:
                self.tr_fields.append(field)

    def avg_time_resolved_collections(self, field: str = None):
        '''
        Average the time resolved data over each collection

        Parameters
        ----------
        field : str
            The field to average. Must be one of self.tr_fields

        Returns
        -------
        avg_tr_collection_data : dict[dict[np.ndarray]]
            The averaged time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field is not None:
            if field not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            # Check if the field has been loaded into self.tr_data. If it unloaded, the list will be empty
            if any([len(self.tr_data[field][key]) == 0 for key in self.tr_data[field]]):
                self.load_time_resolved(field)
            if not hasattr(self, 'avg_tr_collection_data'):
                self.avg_tr_collection_data = {}
            self.avg_tr_collection_data[field] = {}
            for coll in self.tr_data[field]:
                self.avg_tr_collection_data[field][coll] = np.mean(self.tr_data[field][coll], axis=0)
        else:
            self.avg_tr_collection_data = {}
            for fld in self.tr_fields:
                if any([len(self.tr_data[fld][key]) == 0 for key in self.tr_data[fld]]):
                    self.load_time_resolved(fld)
                self.avg_tr_collection_data[fld] = {}
                for coll in self.tr_data[field]:
                    self.avg_tr_collection_data[fld][coll] = np.mean(self.tr_data[fld][coll], axis=0)
        return self.avg_tr_collection_data

    def avg_time_resolved(self, field: str = None):
        '''
        Average the time resolved data over all collections

        Parameters
        ----------
        field : str
            The field to average. Must be one of self.tr_fields

        Returns
        -------
        avg_tr_data : dict[dict[np.ndarray]]
            The averaged time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field is not None:
            if field not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            # Check if the field has been loaded into self.time_resolved_data. If it unloaded, the list will be empty
            if any([len(self.tr_data[field][key]) == 0 for key in self.tr_data[field]]):
                self.load_time_resolved(field)
            if not hasattr(self, 'avg_tr_data'):
                self.avg_tr_data = {}
            self.avg_tr_data[field] = np.mean(np.concatenate([self.tr_data[field][coll] for coll in self.tr_data[field]], axis = 0), axis=0)
        else:
            self.avg_tr_data = {}
            for fld in self.tr_fields:
                if any([len(self.tr_data[fld][key]) == 0 for key in self.tr_data[fld]]):
                    self.load_time_resolved(fld)
                self.avg_tr_data[fld] = np.mean(np.concatenate([self.tr_data[fld][coll] for coll in self.tr_data[fld]], axis = 0), axis=0)
        return self.avg_tr_data

    def avg_time_resolved_over_collections(self, field: str = None):
        '''
        Average the time resolved data over all collections

        Parameters
        ----------
        field : str
            The field to average. Must be one of self.tr_fields

        Returns
        -------
        avg_over_coll_tr_data : dict[dict[np.ndarray]]
            The averaged time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field is not None:
            if field not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            # Check if the field has been loaded into self.time_resolved_data. If it unloaded, the list will be empty
            if any([len(self.tr_data[field][key]) == 0 for key in self.tr_data[field]]):
                self.load_time_resolved(field)

            if not hasattr(self, 'avg_over_coll_tr_data'):
                self.avg_over_coll_tr_data = {}

            # Make sure each time resolved collection data array is the same size
            if not all([np.array_equal(self.tr_data[field][coll][0], self.tr_data[field][coll][1]) for coll in self.tr_data[field]]):
                # If the data is not the same size, get an array of the average data at sligtly adjusted time steps
                tr_dt = self.tr_times[1][1] - self.tr_times[1][0]

                # Get the time in the period of the first timestep of each collection
                tr_coll_start = [self.tr_times[coll][0] % self.interval_period for coll in self.tr_data[field]]

                # If the start times of the collections are all about the same,
                # then take a slice of the first n time steps, where n is the
                # number of time steps in the smallest collection
                # TODO: Make this better. We could look at each start time and see if
                # they things would be better aligned if we used the next timestep
                close_enough = [False] * len(tr_coll_start)
                for ii in range(len(tr_coll_start)):
                    if np.allclose(tr_coll_start, tr_coll_start[ii], atol=self.interval_period/40):#tr_dt/2):
                        close_enough[ii] = True
                if all(close_enough):
                    min_len = min([len(self.tr_data[field][coll]) for coll in self.tr_data[field]])
                    self.avg_over_coll_tr_data[field] = np.stack(np.mean([self.tr_data[field][coll][:min_len] for coll in self.tr_data[field]], axis=0), axis=0)
                else:
                    raise ValueError(f'Start times of the collections are not within tolerance {tr_dt/2} of each other')

            else:
                # Average the data at each time step over each collection. ie. average the data at t=0 over all collections, t=1 over all collections, etc.
                self.avg_over_coll_tr_data[field] = np.stack(np.mean([self.tr_data[field][coll] for coll in self.tr_data[field]], axis=0), axis=0)
        else:
            self.avg_over_coll_tr_data = {}
            for fld in self.tr_fields:
                if any([len(self.tr_data[fld][key]) == 0 for key in self.tr_data[fld]]):
                    self.load_time_resolved(fld)

                # Make sure each time resolved data array is the same size
                if not all([np.array_equal(self.tr_data[fld][coll][0], self.tr_data[fld][coll][1]) for coll in self.tr_data[fld]]):
                    raise ValueError('Time resolved data arrays are not the same size')

                self.avg_over_coll_tr_data[fld] = np.stack(np.mean([self.tr_data[fld][coll] for coll in self.tr_data[fld]], axis=0), axis=0)
        return self.avg_over_coll_tr_data

    def plot_avg_time_resolved_collection(self,
                                          field: str,
                                          collection: int = None,
                                          ax = None,
                                          dpi = 150,
                                          cmap : str = 'GnBu'):
        '''
        Plot the average time resolved data

        Parameters
        ----------
        field : str
            The field to plot
        collection : int, default=None
            The index of the collection to plot. If None, plots all collections
            on a single axis
        ax : matplotlib.axes.Axes, default=None
            The axes object to plot on. If None, creates a new figure and axes
        dpi : int
            The DPI of the plot
        cmap : str, default='GnBu'
            The colormap to use, if plotting multiple collections

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if collection is None:
            return self.plot(field=field, source='tr', show_collections=True,
                             ax=ax, dpi=dpi, cmap=cmap)
        # Single specific collection: ensure data is ready then plot one line
        self._ensure_averaged(field, 'tr')
        edf_type = '_'.join(field.split('_')[:-1]) if field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')) else None
        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots(1, 1, dpi=dpi)
        x, xlabel = self._get_x_data_and_label(
            len(self.avg_tr_collection_data[field][collection]), field, edf_type)
        mid = len(self.tr_times[collection]) // 2
        ax.plot(x, self.avg_tr_collection_data[field][collection],
                label=f't={self.tr_times[collection][mid]:.4e}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self._get_ylabel(field))
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)
        return (fig, ax) if return_fig else ax

    def plot_avg_time_resolved(self, field: str, ax = None, dpi=150):
        '''
        Plot the average time resolved data

        Parameters
        ----------
        field : str
            The field to plot
        ax : matplotlib.axes.Axes, default=None
            The axes object to plot on. If None, creates a new figure and axes
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        return self.plot(field=field, source='tr', show_collections=False,
                         ax=ax, dpi=dpi)

    def animate_time_resolved(self,
                              field: str,
                              collection: int = None,
                              title: str = None,
                              xlabel: str = None,
                              ylabel: str = None,
                              color: str = None,
                              xlim: list[tuple] = None,
                              ylim: list[tuple] = None,
                              normalize: bool = False,
                              log_plot: bool = False,
                              fontsize: int = 12,
                              ticklabelsize: int = 10,
                              dpi=150,
                              frames = None,
                              interval=100,
                              repeat=False,
                              repeat_delay=500
                              ):
        '''
        Animate the time resolved data

        Parameters
        ----------
        field : str
            The field to animate
        collection : int, default=None
            The index of the collection to animate. If None, animates an average
            of all collections. If "full set", animates the full set of collection
            data, concatenated end to end.
        title : str, default=None
            The title of the plot
        xlabel : str, default=None
            The x-axis label
        ylabel : str, default=None
            The y-axis label
        color : str, default=None
            The color of the line. If None, uses black
        xlim : list[tuple], default=None
            The x-axis limits
        ylim : list[tuple], default=None
            The y-axis limits
        fontsize : int, default=12
            The fontsize of the labels
        ticklabelsize : int, default=10
            The fontsize of the tick labels
        dpi : int
            The DPI of the plot
        frames : int, default=None
            The number of frames to animate. If None, animates all frames
        interval : int, default=100
            The interval between frames in milliseconds
        repeat : bool, default=False
            Whether to repeat the animation
        repeat_delay : int, default=1000
            The delay between loops in milliseconds

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field not in self.tr_fields:
            raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
        # Check if the field has been loaded. If unloaded, the list will be empty
        if any([len(self.tr_data[field][key]) == 0 for key in self.tr_data[field]]):
            self.load_time_resolved(field)

        # Set default matplotlib style
        plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': ticklabelsize, 'ytick.labelsize': ticklabelsize})

        fig, ax = plt.subplots(1,1, dpi=dpi)

        edf_type = None
        if field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            edf_type = f"{'_'.join(field.split('_')[:-1])}"

        # Get plot data
        if collection is None:
            if not hasattr(self, 'avg_over_coll_tr_data'):
                self.avg_time_resolved_over_collections(field)
            data = self.avg_over_coll_tr_data[field]
        elif collection == "full set":
            # Get the full set of collection data, concatenated end to end
            data = np.concatenate([self.tr_data[field][coll] for coll in self.tr_data[field]], axis=0)
        else:
            data = self.tr_data[field][collection]

        # Set up plot customizations
        if title is None:
            title = f'Time resolved {field}'
        if ylabel is None:
            ylabel = f'{field}'
        set_xlabel_flag = False
        if xlabel is None:
            set_xlabel_flag = True
        if color is None:
            color = 'black'

        # Get x-axis data
        if set_xlabel_flag:
            x, xlabel = self._get_x_data_and_label(x_length=len(data[0]), field=field, edf_type=edf_type)
        else:
            x, _ = self._get_x_data_and_label(x_length=len(data[0]), field=field, edf_type=edf_type)

        if normalize:
            data = [self._normalize_edf(d, np.diff(x)[i]) for i, d in enumerate(data)]
        if log_plot:
            data = [d / np.abs(x)**0.5 for d in data]

        # Plot initial frame
        line, = ax.plot(x, data[0], color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x=0)
        if log_plot:
            ax.set_yscale('log')

        def update(frame):
            line.set_ydata(data[frame])

            if ylim is None:
                # Get the max and min
                min = np.min(data)
                max = np.max(data)

                ax.set_ylim(min, max)
            else:
                ax.set_ylim(ylim)

            if xlim is not None:
                ax.set_xlim(xlim)

            return line,

        if frames is None:
            frames = len(data)
        if repeat:
            frames *= 2

        anim = FuncAnimation(
            fig,
            update,
            frames = frames,
            interval=interval,
            repeat=repeat,
            repeat_delay=repeat_delay
            )
        return anim

    def animate_time_resolved_grid(self,
                                   field: list[str],
                                   collection: int = None,
                                   title: list[str] = None,
                                   xlabel: list[str] = None,
                                   ylabel: list[str] = None,
                                   color: list[str] = None,
                                   xlim: list[tuple] = None,
                                   ylim: list[tuple] = None,
                                   fontsize: int = 12,
                                   ticklabelsize: int = 10,
                                   dpi=150,
                                   frames: int = None,
                                   interval=100,
                                   repeat=False,
                                   repeat_delay=500
                                   ):
        '''
        Animate the time resolved data

        Parameters
        ----------
        field : list[str]
            The field(s) to animate
        collection : int, default=None
            The index of the collection to animate. If None, animates an average
            of all collections
        title : list[str], default=None
            The title of the plot
        xlabel : list[str], default=None
            The x-axis label
        ylabel : list[str], default=None
            The y-axis label
        color : list[str], default=None
            The color of the line. If None, uses black
        xlim : list[tuple], default=None
            The x-axis limits
        ylim : list[tuple], default=None
            The y-axis limits
        fontsize : int, default=12
            The fontsize of the labels
        ticklabelsize : int, default=10
            The fontsize of the tick labels
        dpi : int
            The DPI of the plot
        frames : int, default=None
            The number of frames to animate. If None, animates all frames
        interval : int, default=100
            The interval between frames in milliseconds
        repeat : bool, default=False
            Whether to repeat the animation
        repeat_delay : int, default=1000
            The delay between loops in milliseconds

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        for fld in field:
            if fld not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            # Check if the field has been loaded. If unloaded, the list will be empty
            if any([len(self.tr_data[fld][key]) == 0 for key in self.tr_data[fld]]):
                self.load_time_resolved(fld)

        # Use the length of the field to determine the number of subplots
        num_plots = len(field)

        edf_type = [None] * len(field)
        for ii, fld in enumerate(field):
            if fld.startswith('EDF'):
                edf_type[ii] = f"{'_'.join(fld.split('_')[:-1])}"

        # Set default matplotlib style
        plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': ticklabelsize, 'ytick.labelsize': ticklabelsize})

        # If 2 field, make them next to each other. If more than 2, make a grid (2x2, 3x3, etc.)
        if num_plots == 1:
            fig, axs = plt.subplots(1,1, dpi=dpi)
        elif num_plots == 2:
            fig, axs = plt.subplots(1,2, dpi=dpi, figsize=(9,3))
        elif num_plots < 5:
            fig, axs = plt.subplots(2, 2, dpi=dpi)
        elif num_plots < 10:
            fig, axs = plt.subplots(3, 3, dpi=dpi)
        elif num_plots < 17:
            fig, axs = plt.subplots(4, 4, dpi=dpi)
        else:
            raise ValueError('Too many fields to plot')

        # Get plot data
        if collection is None:
            data = []
            for fld in field:
                if not hasattr(self, 'avg_over_coll_tr_data'):
                    self.avg_time_resolved_over_collections(fld)
                if fld not in self.avg_over_coll_tr_data:
                    self.avg_time_resolved_over_collections(fld)
                data.append(self.avg_over_coll_tr_data[fld])
        else:
            data = [self.tr_data[fld][collection] for fld in field]

        # Set up plot customizations
        if title is None:
            title = [f'Time resolved {fld}' for fld in field]
        if ylabel is None:
            ylabel = [f'{fld}' for fld in field]
        set_xlabel_flag = False
        if xlabel is None:
            xlabel = []
            set_xlabel_flag = True
        if color is None:
            color = ['black'] * len(field)
        if xlim is None:
            xlim = [None] * len(field)
        if ylim is None:
            ylim = [None] * len(field)

        # Get x-axis data for each field
        x = []
        for ii, fld in enumerate(field):
            x_temp, xlabel_temp = self._get_x_data_and_label(x_length=len(data[ii][0]), field=fld, edf_type=edf_type[ii])
            x.append(x_temp)
            if set_xlabel_flag:
                xlabel.append(xlabel_temp)

        # Plot initial frame
        lines = []
        for ii, ax in enumerate(axs.flat):
            if ii >= len(field):
                ax.axis('off')
                continue

            tmp_line, = ax.plot(x[ii], data[ii][0], color=color[ii])
            lines.append(tmp_line)

            ax.set_xlabel(xlabel[ii])
            ax.set_ylabel(ylabel[ii])
            ax.set_title(title[ii])

            ax.margins(x=0)

            if ylim[ii] is None:
                # Get the max and min
                min = np.min(data[ii])
                max = np.max(data[ii])

                ax.set_ylim(min, max)
            else:
                ax.set_ylim(ylim[ii])

            if xlim[ii] is not None:
                ax.set_xlim(xlim[ii])

        def update(frame):
            for ii, line in enumerate(lines):
                line.set_ydata(data[ii][frame])
            return lines,

        # Get the number of frames
        if frames is None:
            frames = len(data[0])
        if repeat:
            frames *= 2

        if frames > len(data[0]):
            raise ValueError('Number of frames is greater than the number of frames in the data')

        anim = FuncAnimation(
            fig,
            update,
            frames = frames,
            interval=interval,
            repeat=repeat,
            repeat_delay=repeat_delay
            )

        fig.tight_layout()

        return anim


    def animate_time_resolved_grid3(self,
                                   field: list[str],
                                   collection: int = None,
                                   title: list[str] = None,
                                   xlabel: list[str] = None,
                                   ylabel: list[str] = None,
                                   color: list[str] = None,
                                   xlim: list[tuple] = None,
                                   ylim: list[tuple] = None,
                                   normalize: list[bool] = [False]*3,
                                   log_plot: list[bool] = [False]*3,
                                   fontsize: int = 12,
                                   ticklabelsize: int = 10,
                                   dpi=150,
                                   frames: int = None,
                                   interval=100,
                                   repeat=False,
                                   repeat_delay=500
                                   ):
        '''
        Animate the time resolved data

        Parameters
        ----------
        field : list[str]
            The field(s) to animate
        collection : int, default=None
            The index of the collection to animate. If None, animates an average
            of all collections
        title : list[str], default=None
            The title of the plot
        xlabel : list[str], default=None
            The x-axis label
        ylabel : list[str], default=None
            The y-axis label
        color : list[str], default=None
            The color of the line. If None, uses black
        xlim : list[tuple], default=None
            The x-axis limits
        ylim : list[tuple], default=None
            The y-axis limits
        normalize: list[bool], default=[False]*3
            Whether to normalize the EDF
        log_plot: list[bool], default=[False]*3
            Whether to plot the y-axis on a log scale
        fontsize : int, default=12
            The fontsize of the labels
        ticklabelsize : int, default=10
            The fontsize of the tick labels
        dpi : int
            The DPI of the plot
        frames : int, default=None
            The number of frames to animate. If None, animates all frames
        interval : int, default=100
            The interval between frames in milliseconds
        repeat : bool, default=False
            Whether to repeat the animation
        repeat_delay : int, default=1000
            The delay between loops in milliseconds

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        for fld in field:
            if fld not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            # Check if the field has been loaded. If unloaded, the list will be empty
            if any([len(self.tr_data[fld][key]) == 0 for key in self.tr_data[fld]]):
                self.load_time_resolved(fld)

        # Use the length of the field to determine the number of subplots
        num_plots = len(field)

        edf_type = [None] * len(field)
        for ii, fld in enumerate(field):
            if fld.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
                edf_type[ii] = f"{'_'.join(fld.split('_')[:-1])}"

        # Set default matplotlib style
        plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': ticklabelsize, 'ytick.labelsize': ticklabelsize})

        # If 2 field, make them next to each other. If more than 2, make a grid (2x2, 3x3, etc.)
        if num_plots == 1:
            fig, axs = plt.subplots(1,1, dpi=dpi)
        elif num_plots == 2:
            fig, axs = plt.subplots(1,2, dpi=dpi, figsize=(9,4))
        elif num_plots == 3:
            fig, axs = plt.subplots(1,3, dpi=dpi, figsize=(12,4))
        elif num_plots < 5:
            fig, axs = plt.subplots(2, 2, dpi=dpi)
        elif num_plots < 10:
            fig, axs = plt.subplots(3, 3, dpi=dpi)
        elif num_plots < 17:
            fig, axs = plt.subplots(4, 4, dpi=dpi)
        else:
            raise ValueError('Too many fields to plot')

        # Get plot data
        if collection is None:
            data = []
            for fld in field:
                if not hasattr(self, 'avg_over_coll_tr_data'):
                    self.avg_time_resolved_over_collections(fld)
                if fld not in self.avg_over_coll_tr_data:
                    self.avg_time_resolved_over_collections(fld)
                data.append(self.avg_over_coll_tr_data[fld])
        else:
            data = [self.tr_data[fld][collection] for fld in field]

        # Set up plot customizations
        if title is None:
            title = [f'Time resolved {fld}' for fld in field]
        if ylabel is None:
            ylabel = [f'{fld}' for fld in field]
        set_xlabel_flag = False
        if xlabel is None:
            xlabel = []
            set_xlabel_flag = True
        if color is None:
            color = ['black'] * len(field)
        if xlim is None:
            xlim = [None] * len(field)
        if ylim is None:
            ylim = [None] * len(field)

        # Get x-axis data for each field
        x = []
        for ii, fld in enumerate(field):
            x_temp, xlabel_temp = self._get_x_data_and_label(x_length=len(data[ii][0]), field=fld, edf_type=edf_type[ii])
            x.append(x_temp)
            if set_xlabel_flag:
                xlabel.append(xlabel_temp)

        for ii in range(len(data)):
            if normalize[ii]:
                data[ii] = [self._normalize_edf(d, np.diff(x[ii])[i]) for i, d in enumerate(data[ii])]
        for ii in range(len(data)):
            if log_plot[ii]:
                data[ii] = [d / np.abs(x[ii])**0.5 for d in data[ii]]

        # Plot initial frame
        lines = []
        for ii, ax in enumerate(axs.flat):
            if ii >= len(field):
                ax.axis('off')
                continue

            tmp_line, = ax.plot(x[ii], data[ii][0], color=color[ii])
            lines.append(tmp_line)

            ax.set_xlabel(xlabel[ii])
            ax.set_ylabel(ylabel[ii])
            ax.set_title(title[ii])

            ax.margins(x=0)
            if log_plot[ii]:
                ax.set_yscale('log')

            if ylim[ii] is None:
                # Get the max and min
                min = np.min(data[ii])
                max = np.max(data[ii])

                ax.set_ylim(min, max)
            else:
                ax.set_ylim(ylim[ii])

            if xlim[ii] is not None:
                ax.set_xlim(xlim[ii])

        def update(frame):
            for ii, line in enumerate(lines):
                line.set_ydata(data[ii][frame])
            return lines,

        # Get the number of frames
        if frames is None:
            frames = len(data[0])
        if repeat:
            frames *= 2

        if frames > len(data[0]):
            raise ValueError('Number of frames is greater than the number of frames in the data')

        anim = FuncAnimation(
            fig,
            update,
            frames = frames,
            interval=interval,
            repeat=repeat,
            repeat_delay=repeat_delay
            )

        fig.tight_layout()

        return anim

    def integrate_tr_power(self,
                           field: str,
                           collections: bool = False
                           ):
        '''
        Calculate the power from the time resolved data and saves it into
        self.integrated_tr_power[field][coll] with a key coll = 'avg'

        Parameters
        ----------
        field : str
            The field to calculate power for
        collections : int, default=False
            Whether to report power for each collection or as an average
            of all collections

        Returns
        -------
        avg_integrated_tr_power : dict[dict[float]]
            The integrated power for each collection and the average
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field not in ['IPe', 'IPi', 'CPe', 'CPi']:
            raise ValueError('Field must be one of: IPe, IPi, CPe, CPi')

        # Get a profile for each time resolved collection
        if collections:
            if not hasattr(self, 'avg_tr_collection_data'):
                self.avg_time_resolved_collections(field)
            if field not in self.avg_tr_collection_data:
                self.avg_time_resolved_collections(field)

        # Get a profile for the average of all time resolved collections
        if not hasattr(self, 'avg_tr_data'):
            self.avg_time_resolved(field)
        if field not in self.avg_tr_data:
            self.avg_time_resolved(field)

        # Check if the field has been loaded. If unloaded, the list will be empty
        if any([len(self.tr_data[field][key]) == 0 for key in self.tr_data[field]]):
            self.load_time_resolved(field)
        if not hasattr(self, 'avg_integrated_tr_power'):
            self.avg_integrated_tr_power = {}

        self.avg_integrated_tr_power[field] = {}

        # Calculate the power for each collection
        if collections:
            for coll in self.tr_data[field]:
                self.avg_integrated_tr_power[field][coll] = np.sum(self.avg_tr_collection_data[field][coll] * self.dz)

        # Calculate the power for the average of all collections
        self.avg_integrated_tr_power[field]['avg'] = np.sum(self.avg_tr_data[field] * self.dz)

        return self.avg_integrated_tr_power

    def get_integrated_tr_power(self,
                                field: str,
                                collections: bool = False,
                                ):
        '''
        Wrapper function to calculate and display the integrated power

        Parameters
        ----------
        field : str
            The field to calculate power for
        collections : bool, default=False
            Whether to report power for each collection or as an average
            of all collections
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field not in ['IPe', 'IPi', 'CPe', 'CPi']:
            raise ValueError('Field must be one of: IPe, IPi, CPe, CPi')

        # Get the power
        if not hasattr(self, 'integrated_tr_power'):
            self.integrate_tr_power(field, collections=True)
        else:
            if field not in self.avg_integrated_tr_power:
                self.integrate_tr_power(field, collections=True)

        # Display the power
        print(f'Power input from {field}\n--------------------')
        print(f'AVG: {self.avg_integrated_tr_power[field]["avg"]:.3e} W')

        if collections:
            # print a blank line
            print()
            for coll in self.tr_data[field]:
                print(f'{coll:03d}: {self.avg_integrated_tr_power[field][coll]:.3e} W')

    def get_total_tr_power(self):
        '''
        Get the total power into the system from the time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')

        # Look at the list of tr_fields and get power fields
        gather_fields = []
        for fld in self.tr_fields:
            # If anything starts with 'IP' or 'CP', it is a power field
            if fld.startswith('IP'):
                gather_fields.append(fld)
            elif fld.startswith('CP'):
                gather_fields.append(fld)

        # Get the power
        total_power = 0.
        for field in gather_fields:
            self.integrate_tr_power(field)
            total_power += self.avg_integrated_tr_power[field]['avg']

        # Prepare output string
        temp_str = [f'P_{fld[0]},{fld[2]} ({self.avg_integrated_tr_power[fld]["avg"]:.2e})' for fld in gather_fields]
        sum_string = ' + '.join(temp_str)

        # Display the power
        print(f'Total power input\n--------------------')
        print(f' {sum_string} = {total_power:.3e} W')

    def load_time_averaged(self, field: str = None):
        '''
        Load the time averaged data

        Parameters
        ----------
        field : str
            The field to load, if None, loads all fields.

        Returns
        -------
        ta_data : dict[dict[stack of np.ndArray]]
            The time averaged data
        '''
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')

        # Determine fields to load
        if field is not None:
            if field not in self.ta_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.ta_fields)}')
            fields_to_load = [field]
        else:
            fields_to_load = self.ta_fields

        for fld in fields_to_load:
            for coll in self.ta_data[fld]:
                self.ta_data[fld][coll] = np.load(f'{self.ta_colls[coll]}/{fld}.npy')
                if fld.startswith('Jz') and fld != 'Jzc':
                    self.ta_data[fld][coll][0] *= 2
                    self.ta_data[fld][coll][-1] *= 2
        return self.ta_data

    def add_time_averaged_field(self, field: str):
        '''
        Add a time averaged field to the time averaged data

        Parameters
        ----------
        field : str
            The field to add. Must be one of 'P_t', 'EfV', 'Jzc', 'J_t'
        '''
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if field not in ['P_t', 'EfV', 'Jzc', 'J_t']:
            raise ValueError('Field must be one of: P_t, EfV, Jzc, J_t')
        if field == 'P_t':
            self.load_time_averaged('CPe')
            self.load_time_averaged('CPi')
            self.ta_data[field] = {}
            for coll in self.ta_data['CPi']:
                self.ta_data[field][coll] = np.sum((self.ta_data['CPe'][coll] + self.ta_data['CPi'][coll]) * self.dz)
            # Check if the field is already in self.ta_fields before adding
            if field not in self.ta_fields:
                self.ta_fields.append(field)
        elif field == 'EfV':
            self.load_time_averaged('phi')
            self.ta_data[field] = {}
            for coll in self.ta_data['phi']:
                self.ta_data[field][coll] = -np.gradient(self.ta_data['phi'][coll], self.dz)
            # Check if the field is already in self.ta_fields before adding
            if field not in self.ta_fields:
                self.ta_fields.append(field)
        elif field == 'Jzc':
            self.load_time_averaged('Jze')
            self.load_time_averaged('Jzi')
            self.ta_data[field] = {}
            for coll in self.ta_data['Jze']:
                self.ta_data[field][coll] = self.ta_data['Jze'][coll] + self.ta_data['Jzi'][coll]
            # Check if the field is already in self.ta_fields before adding
            if field not in self.ta_fields:
                self.ta_fields.append(field)
        elif field == 'J_t':
            self.load_time_averaged('Jze')
            self.load_time_averaged('Jzi')
            self.load_time_averaged('J_d')
            self.ta_data[field] = {}
            for coll in self.ta_data['Jze']:
                # Interpolate J_d from cells to nodes
                J_d_on_nodes = np.interp(self.nodes, self.cells, self.ta_data['J_d'][coll])
                self.ta_data[field][coll] = self.ta_data['Jze'][coll] + self.ta_data['Jzi'][coll] + J_d_on_nodes
            # Check if the field is already in self.ta_fields before adding
            if field not in self.ta_fields:
                self.ta_fields.append(field)

    def avg_time_averaged(self, field: str = None):
        '''
        Average the time averaged data over all collections

        Parameters
        ----------
        field : str
            The field to average. Must be one of self.ta_fields

        Returns
        -------
        avg_ta_data : dict[np.ndarray]
            The averaged time averaged data
        '''
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if field is not None:
            if field not in self.ta_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.ta_fields)}')
            # Check if the field has been loaded into self.ta_data. If it unloaded, the list will be empty
            if field == 'P_t':
                pass  # P_t is a special case, loaded in add_time_averaged_field
            elif any([len(self.ta_data[field][key]) == 0 for key in self.ta_data[field]]):
                self.load_time_averaged(field)
            if not hasattr(self, 'avg_ta_data'):
                self.avg_ta_data = {}
            self.avg_ta_data[field] = np.mean([self.ta_data[field][coll] for coll in self.ta_data[field]], axis=0)
        else:
            # When no field is specified, refresh ta_fields to include all available files
            # This ensures we capture any fields that might have been missed during initialization
            all_available_fields = [file.split('.')[0] for file in os.listdir(self.ta_colls[1]) if file.endswith('.npy')]
            all_available_fields.sort()

            # Update ta_fields and ta_data structure for any new fields
            for fld in all_available_fields:
                if fld not in self.ta_fields:
                    self.ta_fields.append(fld)
                    self.ta_data[fld] = {}
                    for collection in self.ta_colls:
                        self.ta_data[fld][collection] = []

            self.avg_ta_data = {}
            for fld in all_available_fields:
                if any([len(self.ta_data[fld][key]) == 0 for key in self.ta_data[fld]]):
                    self.load_time_averaged(fld)
                self.avg_ta_data[fld] = np.mean([self.ta_data[fld][coll] for coll in self.ta_data[fld]], axis=0)

    def avg_intervals_over_time(self, field: str = None):
        '''
        Average interval data over all collection averaged slices to
        create a collection-averaged time-averaged profile. Array is
        saved to self.avg_time_avg_in_data[field]

        Parameters
        ----------
        field : str, default=None
            Field to average over time. If None, all fields are averaged.
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        # Average interval data if not already done
        if not hasattr(self, 'avg_in_data'):
            self.avg_intervals(field)
        if field not in self.avg_in_data:
            self.avg_intervals(field)

        # Create attribute to store time-averaged interval data if it doesn't exist
        if not hasattr(self, 'avg_time_avg_in_data'):
            self.avg_time_avg_in_data = {}

        # Get list of fields to average
        fields = [field] if field is not None else self.in_fields

        for f in fields:
            self.avg_time_avg_in_data[f] = np.zeros_like(self.avg_in_data[f][0])
            for ii in range(len(self.in_times)):
                self.avg_time_avg_in_data[f] += self.avg_in_data[f][ii] / len(self.in_times)

    def avg_intervals_by_collection_over_time(self, field: str):
        '''
        Average interval data for each collection over all slices to
        create a time-averaged profile for each collection. Array is
        saved to self.time_avg_in_data[field][diag_collection]

        Parameters
        ----------
        field : str
            Field to get fully averaged data for
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        # Check if field has been loaded into self.interval_data
        if any([np.array_equal(self.in_data[field][coll][0], 0) for coll in self.in_data[field]]):
            self.load_intervals(field)

        # Create attribute to store time-averaged interval collection data if it doesn't exist
        if not hasattr(self, 'time_avg_in_data'):
            self.time_avg_in_data = {}

        # Get list of fields to average
        fields = [field] if field is not None else self.in_fields
        for f in fields:
            # Make a dictionary where each key is a diagnostic collection
            self.time_avg_in_data[f] = {}
            for coll in self.in_colls:
                # Each dictionary item is an array of the interval slices time averaged
                self.time_avg_in_data[f][coll] = np.zeros_like(self.in_data[f][coll][0])
                for ii in range(len(self.in_times)):
                    self.time_avg_in_data[f][coll] += self.in_data[f][coll][ii] / len(self.in_times)

    def plot_intervals_time_averaged(self,
                                    field: str,
                                    plot_all_coll: bool = True,
                                    custom_avg_label = None,
                                    custom_avg_color = None,
                                    edf_log_plot = False,
                                    ax = None,
                                    dpi: int = 150,
                                    cmap: str = 'GnBu'):
        '''
        Plot time-averaged interval data for a field

        Parameters
        ----------
        field : str
            Field to plot
        plot_all_coll : bool, default=True
            Plot all collections in the background if True
        custom_avg_label : str, default='Average'
            The label to use for the average line
        custom_avg_color : str, default=None
            The color to use for the average line
        edf_log_plot : bool, default=False
            Whether to plot the EDF in log scale on the y-axis
        ax : matplotlib.axes.Axes, default=None
            Axes to plot on. If None, creates a new figure and axes.
        dpi : int, default=150
            DPI of the plot if creating a new figure
        cmap : str, default='GnBu'
            Colormap to use for collections

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if edf_log_plot and not field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            raise ValueError('Field must be an EDF')
        return self.plot(field=field, source='in', show_collections=plot_all_coll,
                         log_scale=edf_log_plot, custom_label=custom_avg_label,
                         custom_color=custom_avg_color, ax=ax, dpi=dpi, cmap=cmap)

    def plot_time_averaged(self,
                           field: str,
                           plot_all_coll = True,
                           custom_avg_label = None,
                           custom_avg_color = None,
                           edf_log_plot = False,
                           ax = None,
                           dpi=150,
                           cmap = 'coolwarm'):
        '''
        Plot the time averaged data

        Parameters
        ----------
        field : str
            The field to plot
        plot_all_coll : bool, default=True
            Whether to plot all collections on the same axis
        custom_avg_label : str, default='Average'
            The label to use for the average line
        custom_avg_color : str, default=None
            The color to use for the average line
        edf_log_plot : bool, default=False
            Whether to plot the EDF in log scale on the y-axis
        ax : matplotlib.axes.Axes, default=None
            The axes object to plot on. If None, creates a new figure and axes
        dpi : int

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if edf_log_plot and not field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            raise ValueError('Field must be an EDF')
        return self.plot(field=field, source='ta', show_collections=plot_all_coll,
                         log_scale=edf_log_plot, custom_label=custom_avg_label,
                         custom_color=custom_avg_color, ax=ax, dpi=dpi, cmap=cmap)

    def plot_time_averaged_distributions(self,
                                         species: str,
                                         dir: str = '',
                                         multiple: int = 1,
                                         normalize = True,
                                         log_plot = True,
                                         ax = None,
                                         dpi=150,
                                         cmap = 'managua'):
        '''
        Plot each of the time averaged EDFs for a given species and direction on the same axis

        Parameters
        ----------
        species : str
            The species to plot
        dir : str, default=''
            The direction of the distribution to plot (e.g., 'x', 'y', 'z', or'' for total)
        multiple : int, default=1
            Only plot EDFs that are a multiple of this number (e.g., 10 to plot EDFs 1, 10, 20, etc.)
        normalize : bool, default=True
            Whether to normalize the distributions (for EDFs)
        log_plot : bool, default=True
            Whether to plot the distribution in log scale on the y-axis
        ax : matplotlib.axes.Axes, default=None
            The axes object to plot on. If None, creates a new figure and axes
        dpi : int

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        base_field = f'E{dir}DF_{species}' if dir else f'EDF_{species}'
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if not base_field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            raise ValueError('Field must be an EDF')
        if f'{base_field}_01' not in self.ta_fields:
             raise ValueError(f'Species and dir must come from: {[fld for fld in self.ta_fields if fld.startswith("E")]}')

        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            return_fig = True

        for edf_idx in range(1, self.num_edfs + 1):
            # Continue if not a multiple of 10 and not 1 (e.g., 1, 10, 20, etc.)
            if edf_idx != 1 and edf_idx % multiple != 0:
                continue
            field = f'{base_field}_{edf_idx:02d}'
            # Load the field
            if any([len(self.ta_data[field][key]) == 0 for key in self.ta_data[field]]):
                self.load_time_averaged(field)
            # Make avg line
            if not hasattr(self, 'avg_ta_data'):
                self.avg_time_averaged(field)
            if field not in self.avg_ta_data:
                self.avg_time_averaged(field)

            # Set plot labels
            x, xlabel = self._get_x_data_and_label(x_length=len(self.avg_ta_data[field]), field=field, edf_type=base_field)
            color = self._color_chooser(edf_idx, self.num_edfs, cmap=cmap, reverse=True)
            label = f'{base_field.split("_")[0]} {edf_idx}'

            data = self.avg_ta_data[field]
            if normalize:
                dE = np.diff(x)[0]
                data = self._normalize_edf(data, dE)
            if log_plot:
                data /= np.abs(x)**0.5

            ax.plot(x, data, label=label, color=color, linewidth=2, linestyle='solid')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.ylabel_dict[field.split('_')[0]])
        ax.set_title(f'Comparison of Time Averaged {species} E{dir}DFs')
        ax.margins(x=0)
        ax.legend(fontsize = 'small')
        if log_plot:
            ax.set_yscale('log')
        plt.tight_layout()

        if not return_fig:
            return ax
        return fig, ax

    def plot(self,
             field: str,
             source: str = None,
             show_collections: bool = True,
             normalize: bool = False,
             log_scale: bool = False,
             custom_label: str = None,
             custom_color: str = None,
             ax = None,
             dpi: int = 150,
             cmap: str = 'GnBu'):
        '''
        Plot collection-averaged field data from any data source.

        This unified method replaces plot_time_averaged, plot_avg_time_resolved,
        and plot_intervals_time_averaged. The source is auto-detected by default
        (preferring ta > tr > in) but can be specified explicitly.

        Parameters
        ----------
        field : str
            The field to plot.
        source : str, default=None
            Data source: 'ta' (time-averaged), 'tr' (time-resolved), or 'in'
            (interval). If None, auto-detects with priority ta > tr > in.
        show_collections : bool, default=True
            Overlay per-collection background lines to show convergence.
            For 'ta' and 'in' sources: each diagnostic collection is one line.
            For 'tr' source: each line is the time-average of one collection.
        normalize : bool, default=False
            Normalize EDF data so the integral over energy equals 1.
        log_scale : bool, default=False
            Use a log y-axis. For EDF fields this also applies the Druyvesteyn
            transform (divides by sqrt(energy)) before plotting.
        custom_label : str, default=None
            Override the average-line label. Useful when overlaying multiple
            datasets on the same axes.
        custom_color : str, default=None
            Override the average-line color (default black).
        ax : matplotlib.axes.Axes, default=None
            Axes to plot on. If None, a new figure and axes are created and
            (fig, ax) is returned. If provided, only ax is returned.
        dpi : int, default=150
            DPI for new figures.
        cmap : str, default='GnBu'
            Colormap for collection background lines.

        Returns
        -------
        (fig, ax) if ax was None, else ax.
        '''
        if source is None:
            source = self._auto_detect_source(field)
        self._ensure_averaged(field, source)

        edf_type = None
        if field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            edf_type = '_'.join(field.split('_')[:-1])

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots(1, 1, dpi=dpi)

        # Select the fully-averaged data array and x-axis
        if source == 'ta':
            avg_data = self.avg_ta_data[field]
        elif source == 'tr':
            avg_data = self.avg_tr_data[field]
        else:
            avg_data = self.avg_time_avg_in_data[field]

        x, xlabel = self._get_x_data_and_label(len(avg_data), field, edf_type)

        def _apply_transforms(data):
            data = data.copy()
            if normalize:
                data = self._normalize_edf(data, np.diff(x)[0])
            if log_scale:
                data = data / np.abs(x) ** 0.5
            return data

        # Background per-collection lines
        if show_collections:
            if source == 'ta':
                coll_data = self.ta_data[field]
            elif source == 'tr':
                coll_data = self.avg_tr_collection_data[field]
            else:
                coll_data = self.time_avg_in_data[field]
            num = len(coll_data)
            for coll in coll_data:
                ax.plot(x, _apply_transforms(coll_data[coll]),
                        label=f'Collection {coll}', alpha=0.4,
                        color=self._color_chooser(coll, num, cmap=cmap))

        # Average line — cycle linestyle when called multiple times on the same axes
        add_legend = False
        if not return_fig and not show_collections:
            num_avg = len([l for l in ax.lines if l.get_label().startswith('Average')])
            styles = ['solid', 'dotted', 'dashdot', 'dashed']
            avg_linestyle = styles[num_avg % len(styles)]
            avg_label = f'Average ({num_avg + 1})'
            add_legend = True
            for line in ax.lines:
                if custom_label is None and line.get_label() == 'Average':
                    line.set_label('Average (1)')
                    break
        else:
            avg_linestyle = 'solid'
            avg_label = 'Average'

        if custom_label is not None:
            avg_label = custom_label
        avg_color = custom_color if custom_color is not None else 'black'

        ax.plot(x, _apply_transforms(avg_data),
                label=avg_label, color=avg_color, linewidth=2, linestyle=avg_linestyle)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(self._get_ylabel(field))
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)
        if show_collections or add_legend:
            ax.legend(fontsize='small')
        if log_scale:
            ax.set_yscale('log')

        return (fig, ax) if return_fig else ax

    def plot_phase_resolved(self,
                            field: str,
                            interval: int = None,
                            plot_time_avg: bool = True,
                            ax = None,
                            dpi: int = 150,
                            cmap: str = 'GnBu'):
        '''
        Plot phase-resolved interval data, showing the field at each time
        slice within the RF period.

        Use this instead of plot() with source='in' when you want to see the
        RF phase structure rather than the convergence across collections.

        Parameters
        ----------
        field : str
            The field to plot.
        interval : int, default=None
            Index (0 to len(in_times)-1) of the specific phase slice to plot.
            If None, all phase slices are plotted on a single axis.
        plot_time_avg : bool, default=True
            Overlay a time-averaged line. Prefers ta > tr > interval average.
        ax : matplotlib.axes.Axes, default=None
            Axes to plot on. If None, a new figure and axes are created.
        dpi : int, default=150
            DPI for new figures.
        cmap : str, default='GnBu'
            Colormap for the phase-slice lines.

        Returns
        -------
        (fig, ax) if ax was None, else ax.
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        if field not in self.in_fields:
            raise ValueError(f'Field must be one of: {", ".join(self.in_fields)}')

        if not hasattr(self, 'avg_in_data') or field not in self.avg_in_data:
            self.avg_intervals(field)

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots(1, 1, dpi=dpi)

        edf_type = None
        if field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            edf_type = '_'.join(field.split('_')[:-1])

        x, xlabel = self._get_x_data_and_label(
            len(self.avg_in_data[field][0]), field, edf_type)

        # Resolve the best available time-averaged reference line (ta > tr > in)
        avg_line = None
        if plot_time_avg:
            if getattr(self, 'ta_bool', False):
                try:
                    if not hasattr(self, 'avg_ta_data') or field not in self.avg_ta_data:
                        self.avg_time_averaged(field)
                    avg_line = self.avg_ta_data.get(field)
                except ValueError:
                    pass
            if avg_line is None and getattr(self, 'tr_bool', False):
                try:
                    if not hasattr(self, 'avg_tr_data') or field not in self.avg_tr_data:
                        self.avg_time_resolved(field)
                    avg_line = self.avg_tr_data.get(field)
                except ValueError:
                    pass
            if avg_line is None:
                try:
                    if not hasattr(self, 'avg_time_avg_in_data') or field not in self.avg_time_avg_in_data:
                        self.avg_intervals_over_time(field)
                    avg_line = self.avg_time_avg_in_data.get(field)
                except ValueError:
                    pass
            if avg_line is None:
                print(f'Warning: no time-averaged data found for {field!r}; skipping avg line.')

        if interval is None:
            num = len(self.in_times)
            for ii in range(num):
                ax.plot(x, self.avg_in_data[field][ii],
                        label=f't={self.in_times[ii]:.3f}*T',
                        color=self._color_chooser(ii, num, cmap=cmap))
            if avg_line is not None:
                ax.plot(x, avg_line, label='Average', color='black')
            ax.set_title(f'{field} phase-resolved')
            ax.legend(loc=[1.01, 0], fontsize='small')
        else:
            ax.plot(x, self.avg_in_data[field][interval],
                    label=f't={self.in_times[interval]:.3f}*T',
                    color=self._color_chooser(interval, len(self.in_times), cmap=cmap))
            ax.set_title(f'{field} at t = {self.in_times[interval]:.3f}*T')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(self._get_ylabel(field))
        ax.margins(x=0)

        return (fig, ax) if return_fig else ax

    def calculate_time_averaged_rates(self):
        ''''''
        pass

    def _import_cross_sections(self):
        '''Import references to the cross section files from warpx_used_inputs file in the diagnostics directory'''
        warpx_inputs_file = os.path.join(self.directory, 'warpx_used_inputs')
        if not os.path.exists(warpx_inputs_file):
            raise FileNotFoundError(f"warpx_used_inputs file not found in {self.directory}")

        # Dictionary to store cross section data
        self.cross_section_dict = {}

        # Parse the warpx_used_inputs file
        with open(warpx_inputs_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Look for cross section entries
                if '_cross_section' in line and '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        process_key = parts[0].strip()
                        file_path = parts[1].strip()

                        # Extract the process name from the filename, not the key
                        # e.g., "/path/to/e_momentumTransfer.dat" -> "e_momentumTransfer"
                        if '.' in process_key:
                            warpx_process_name = process_key.split('.')[-1].replace('_cross_section', '')

                            # Load the cross section data
                            try:
                                data = np.loadtxt(file_path)
                                energy = data[:, 0]  # Energy in eV
                                cross_section = data[:, 1]  # Cross section in m²

                                # Extract filename without path and extension for naming
                                filename = os.path.basename(file_path)
                                if filename.endswith('.dat'):
                                    filename = filename[:-4]

                                self.cross_section_dict[filename] = {
                                    'energy': energy,
                                    'cross_section': cross_section,
                                    'warpx_process_type': warpx_process_name,
                                    'file_path': file_path
                                }
                            except Exception as e:
                                print(f"Warning: Could not load cross section file {file_path}: {e}")

        print(f"Loaded {len(self.cross_section_dict)} cross section files:")
        for name in self.cross_section_dict.keys():
            print(f"  - {name}")

    def _get_velocity_from_energy(self, energy):
        """Convert energy in eV to velocity in m/s."""
        # Energy in eV to Joules
        energy_joules = energy * 1.60218e-19  # 1 eV = 1.60218e-19 J
        # v = sqrt(2 * E / m), where m is the mass of an electron in kg
        m_electron = 9.10938e-31  # kg
        velocity = np.sqrt(2 * energy_joules / m_electron)
        return velocity

    def _interpolate_and_extrapolate_cross_section(self, eedf_energy, cross_section_energy, cross_section_data):
        """
        Interpolate cross section data to match the energy bins.
        Uses np.interp with default extrapolation (constant values at boundaries).
        """
        return np.interp(eedf_energy, cross_section_energy, cross_section_data)

    def _normalize_edf(self, edf, dE):
        """Normalize EEDF data to ensure the integral over energy is 1."""
        integral = np.sum(edf * dE)
        if integral <= 0:
            raise ValueError("Integral of EDF is zero, cannot normalize.")
        return edf / integral

    def calculate_reaction_rate_coefficients(self, verbose=False):
        """
        Calculate reaction rate coefficients from EEDFs and cross sections.

        Returns
        -------
        dict
            Dictionary with reaction names as keys and rate coefficients as values
        """
        # Check if EEDFs are available
        if not hasattr(self, 'edf_energy') or f'EDF_{self.species_names[0]}' not in self.edf_energy:
            raise ValueError("EEDFs not found. Make sure EEDFs are available in the diagnostics.")

        # Import cross sections if not already done
        if not hasattr(self, 'cross_section_dict'):
            self._import_cross_sections()

        # Get time-averaged EEDF data
        self.avg_time_averaged()

        # Get EEDF fields and sort them numerically
        eedf_fields = [field for field in self.avg_ta_data.keys() if field.startswith(f'EDF_{self.species_names[0]}')]
        # Sort numerically by the box number to ensure proper spatial ordering
        eedf_fields.sort(key=lambda x: int(x.split('_')[2]) if '_' in x else 0)
        if not eedf_fields:
            raise ValueError("No EEDF data found in time-averaged diagnostics.")

        # Energy bins and spacing
        energy_bins = self.edf_energy[f'EDF_{self.species_names[0]}']
        dE = np.diff(energy_bins)[0]  # Assuming uniform spacing

        # Initialize rate coefficient storage
        self.rate_coefficients = {}  # k values (m³/s)
        self.reaction_rates = {}     # k×N_e values (1/s)

        # Get electron density data if available
        electron_density = None
        if 'N_e' in self.avg_ta_data:
            electron_density = self.avg_ta_data['N_e']

        # Calculate rate coefficients for each reaction
        for reaction_name, cross_section_info in self.cross_section_dict.items():
            cross_section_energy = cross_section_info['energy']
            cross_section_data = cross_section_info['cross_section']

            if verbose:
                print(f"Calculating rate coefficients for {reaction_name}...")

            # Initialize arrays for rate coefficients across all EEDF regions
            rate_coeffs_by_region = []

            # Process each EEDF field (spatial region)
            for field in eedf_fields:
                eedf_data = self.avg_ta_data[field]

                # Normalize EEDF
                integral = np.sum(eedf_data * dE)
                if integral > 0:
                    eedf_normalized = eedf_data / integral
                else:
                    eedf_normalized = np.zeros_like(eedf_data)

                # Interpolate cross sections to EEDF energy grid
                interpolated_cross_section = self._interpolate_and_extrapolate_cross_section(
                    energy_bins, cross_section_energy, cross_section_data)

                # Calculate velocities
                velocities = self._get_velocity_from_energy(energy_bins)

                # Calculate rate coefficient: k = ∫ σ(E) * v(E) * EEDF(E) * dE
                rate_coefficient = np.sum(interpolated_cross_section * velocities * eedf_normalized * dE)
                rate_coeffs_by_region.append(rate_coefficient)

            # Store rate coefficients (k values)
            self.rate_coefficients[reaction_name] = np.array(rate_coeffs_by_region)

            # Calculate reaction rates (k×N_e) if electron density is available
            if electron_density is not None:
                # Interpolate electron density to EDF box positions
                if len(electron_density) == len(self.cells):
                    # Electron density is on cells, need to average over EDF boxes
                    reaction_rates_by_region = []
                    for i in range(self.num_edfs):
                        # Find cell indices for this EDF box
                        start_idx = self.edf_boundary_node_indices[i]
                        end_idx = self.edf_boundary_node_indices[i + 1]
                        # Average electron density in this region
                        avg_density = np.mean(electron_density[start_idx:end_idx])
                        reaction_rate = rate_coeffs_by_region[i] * avg_density
                        reaction_rates_by_region.append(reaction_rate)
                    self.reaction_rates[reaction_name] = np.array(reaction_rates_by_region)
                elif len(electron_density) == len(self.nodes):
                    # Electron density is on nodes, need to average over EDF boxes
                    reaction_rates_by_region = []
                    for i in range(self.num_edfs):
                        # Find node indices for this EDF box
                        start_idx = self.edf_boundary_node_indices[i]
                        end_idx = self.edf_boundary_node_indices[i + 1]
                        # Average electron density in this region
                        avg_density = np.mean(electron_density[start_idx:end_idx])
                        reaction_rate = rate_coeffs_by_region[i] * avg_density
                        reaction_rates_by_region.append(reaction_rate)
                    self.reaction_rates[reaction_name] = np.array(reaction_rates_by_region)
                elif len(electron_density) == len(rate_coeffs_by_region):
                    # Direct multiplication if dimensions match (electron density already per EDF box)
                    self.reaction_rates[reaction_name] = self.rate_coefficients[reaction_name] * electron_density
                else:
                    # Try to interpolate electron density to EDF box positions
                    print(f"Warning: Electron density shape {electron_density.shape} doesn't match expected shapes. Skipping reaction rates.")
                    break

        if verbose:
            print(f"Calculated rate coefficients for {len(self.rate_coefficients)} reactions")
            print(f"Reactions: {list(self.rate_coefficients.keys())}")

        return self.rate_coefficients

    def plot_rate_coefficients(self, reaction_name=None, ax=None, dpi=150, **kwargs):
        """
        Plot reaction rate coefficients vs position.

        Parameters
        ----------
        reaction_name : str, optional
            Name of the reaction to plot. If None, plots all reactions on separate subplots.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        dpi : int
            DPI for the figure.
        **kwargs
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axes objects.
        """
        if not hasattr(self, 'rate_coefficients'):
            raise ValueError("Rate coefficients not calculated. Run calculate_reaction_rate_coefficients() first.")

        # Get positions for EDF boxes (midpoints)
        if hasattr(self, 'edf_box_boundaries'):
            positions = []
            for i in range(self.num_edfs):
                mid_pos = (self.edf_box_boundaries[i] + self.edf_box_boundaries[i + 1]) / 2
                positions.append(mid_pos)
            positions = np.array(positions) * 1000  # Convert to mm
        else:
            positions = np.arange(len(list(self.rate_coefficients.values())[0]))

        return_fig = False
        if ax is None:
            return_fig = True
            if reaction_name is None:
                # Create subplots for all reactions
                n_reactions = len(self.rate_coefficients)
                ncols = min(3, n_reactions)
                nrows = (n_reactions + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=dpi)
                if n_reactions == 1:
                    axes = [axes]
                elif nrows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                for i, (name, coeffs) in enumerate(self.rate_coefficients.items()):
                    if i < len(axes):
                        axes[i].plot(positions, coeffs, **kwargs)
                        axes[i].set_title(name)
                        axes[i].set_xlabel('Position [mm]')
                        axes[i].set_ylabel('Rate Coefficient [m³/s]')
                        axes[i].grid(True, alpha=0.3)
                        axes[i].margins(x=0)

                # Hide unused subplots
                for i in range(n_reactions, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                return fig, axes
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

        # Plot single reaction
        if reaction_name is None:
            reaction_name = list(self.rate_coefficients.keys())[0]

        if reaction_name not in self.rate_coefficients:
            raise ValueError(f"Reaction '{reaction_name}' not found. Available reactions: {list(self.rate_coefficients.keys())}")

        coeffs = self.rate_coefficients[reaction_name]
        ax.plot(positions, coeffs, **kwargs)
        ax.set_title(f'Rate Coefficient: {reaction_name}')
        ax.set_xlabel('Position [mm]')
        ax.set_ylabel('Rate Coefficient [m³/s]')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

        if return_fig:
            return fig, ax
        else:
            return ax

    def plot_reaction_rates(self, reaction_name=None, ax=None, dpi=150, **kwargs):
        """
        Plot reaction rates vs position.

        Parameters
        ----------
        reaction_name : str, optional
            Name of the reaction to plot. If None, plots all reactions on separate subplots.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        dpi : int
            DPI for the figure.
        **kwargs
            Additional keyword arguments passed to the plot function.

        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axes objects.
        """
        if not hasattr(self, 'reaction_rates'):
            raise ValueError("Reaction rates not calculated. Run calculate_reaction_rate_coefficients() first.")

        if len(self.reaction_rates) == 0:
            raise ValueError("No reaction rates calculated. Electron density may not be available.")

        # Get positions for EDF boxes (midpoints)
        if hasattr(self, 'edf_box_boundaries'):
            positions = []
            for i in range(self.num_edfs):
                mid_pos = (self.edf_box_boundaries[i] + self.edf_box_boundaries[i + 1]) / 2
                positions.append(mid_pos)
            positions = np.array(positions) * 1000  # Convert to mm
        else:
            positions = np.arange(len(list(self.reaction_rates.values())[0]))

        return_fig = False
        if ax is None:
            return_fig = True
            if reaction_name is None:
                # Create subplots for all reactions
                n_reactions = len(self.reaction_rates)
                ncols = min(3, n_reactions)
                nrows = (n_reactions + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=dpi)
                if n_reactions == 1:
                    axes = [axes]
                elif nrows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                for i, (name, rates) in enumerate(self.reaction_rates.items()):
                    if i < len(axes):
                        axes[i].plot(positions, rates, **kwargs)
                        axes[i].set_title(name)
                        axes[i].set_xlabel('Position [mm]')
                        axes[i].set_ylabel('Reaction Rate [1/s]')
                        axes[i].grid(True, alpha=0.3)
                        axes[i].margins(x=0)

                # Hide unused subplots
                for i in range(n_reactions, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                return fig, axes
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

        # Plot single reaction
        if reaction_name is None:
            reaction_name = list(self.reaction_rates.keys())[0]

        if reaction_name not in self.reaction_rates:
            raise ValueError(f"Reaction '{reaction_name}' not found. Available reactions: {list(self.reaction_rates.keys())}")

        rates = self.reaction_rates[reaction_name]
        ax.plot(positions, rates, **kwargs)
        ax.set_title(f'Reaction Rate: {reaction_name}')
        ax.set_xlabel('Position [mm]')
        ax.set_ylabel('Reaction Rate [1/s]')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

        if return_fig:
            return fig, ax
        else:
            return ax

    def get_available_reactions(self):
        """
        Get list of available reactions.

        Returns
        -------
        list
            List of reaction names.
        """
        if hasattr(self, 'cross_section_dict'):
            return list(self.cross_section_dict.keys())
        else:
            self._import_cross_sections()
            return list(self.cross_section_dict.keys())

    def get_edf_box_positions(self):
        """
        Get the midpoint positions of EDF boxes.

        Returns
        -------
        np.ndarray
            Array of EDF box midpoint positions in meters.
        """
        if hasattr(self, 'edf_box_boundaries'):
            positions = []
            for i in range(self.num_edfs):
                mid_pos = (self.edf_box_boundaries[i] + self.edf_box_boundaries[i + 1]) / 2
                positions.append(mid_pos)
            return np.array(positions)
        else:
            raise ValueError("EDF box boundaries not found.")

    def get_rate_coefficient_summary(self):
        """
        Print a summary of calculated rate coefficients.
        """
        if not hasattr(self, 'rate_coefficients'):
            print("No rate coefficients calculated. Run calculate_reaction_rate_coefficients() first.")
            return

        print("Rate Coefficient Summary:")
        print("=" * 50)

        for reaction_name, coeffs in self.rate_coefficients.items():
            mean_coeff = np.mean(coeffs)
            max_coeff = np.max(coeffs)
            min_coeff = np.min(coeffs)

            print(f"{reaction_name}:")
            print(f"  Mean: {mean_coeff:.2e} m³/s")
            print(f"  Max:  {max_coeff:.2e} m³/s")
            print(f"  Min:  {min_coeff:.2e} m³/s")
            print()

        if hasattr(self, 'reaction_rates') and len(self.reaction_rates) > 0:
            print("Reaction Rate Summary:")
            print("=" * 50)

            for reaction_name, rates in self.reaction_rates.items():
                mean_rate = np.mean(rates)
                max_rate = np.max(rates)
                min_rate = np.min(rates)

                print(f"{reaction_name}:")
                print(f"  Mean: {mean_rate:.2e} 1/s")
                print(f"  Max:  {max_rate:.2e} 1/s")
                print(f"  Min:  {min_rate:.2e} 1/s")
                print()

    def _auto_detect_source(self, field: str) -> str:
        '''Auto-detect which data source (ta/tr/in) contains the given field.

        Priority order: time-averaged (ta) > time-resolved (tr) > interval (in),
        as ta represents the most converged representation when available.

        Parameters
        ----------
        field : str
            The field name to look up.

        Returns
        -------
        str
            One of 'ta', 'tr', or 'in'.
        '''
        if self.ta_bool and field in self.ta_fields:
            return 'ta'
        if self.tr_bool and field in self.tr_fields:
            return 'tr'
        if self.in_bool and field in self.in_fields:
            return 'in'
        avail = (f'ta={getattr(self, "ta_fields", [])} '
                 f'tr={getattr(self, "tr_fields", [])} '
                 f'in={getattr(self, "in_fields", [])}')
        raise ValueError(f'Field {field!r} not found in any source. Available: {avail}')

    def _get_ylabel(self, field: str) -> str:
        '''Return the y-axis label for a field via ylabel_dict.

        Parameters
        ----------
        field : str
            The field name.

        Returns
        -------
        str
            The y-axis label string.
        '''
        if field.startswith(('EDF', 'ExDF', 'EyDF', 'EzDF')):
            diag_type = field.split('_')[0]
        elif field.startswith(('P_I', 'P_C', 'J_d', 'E_z')):
            diag_type = '_'.join(field.split('_')[:2])
        else:
            diag_type = field.split('_')[0]
        return self.ylabel_dict.get(diag_type, field)

    def _ensure_averaged(self, field: str, source: str):
        '''Ensure data for field is loaded and all averages are computed.

        For 'ta': loads raw collection data and computes avg_ta_data.
        For 'tr': loads raw data, computes avg_tr_data (full average) and
                  avg_tr_collection_data (per-collection time-average).
        For 'in': loads raw data, computes avg_time_avg_in_data (time- and
                  collection-averaged) and time_avg_in_data (per-collection
                  time-averaged).

        Parameters
        ----------
        field : str
            The field name.
        source : str
            One of 'ta', 'tr', or 'in'.
        '''
        if source == 'ta':
            if not self.ta_bool:
                raise ValueError('Time averaged data not found')
            if field not in self.ta_fields:
                raise ValueError(f'Field {field!r} not in time-averaged fields: {self.ta_fields}')
            if any(len(self.ta_data[field][k]) == 0 for k in self.ta_data[field]):
                self.load_time_averaged(field)
            if not hasattr(self, 'avg_ta_data') or field not in self.avg_ta_data:
                self.avg_time_averaged(field)

        elif source == 'tr':
            if not self.tr_bool:
                raise ValueError('Time resolved data not found')
            if field not in self.tr_fields:
                raise ValueError(f'Field {field!r} not in time-resolved fields: {self.tr_fields}')
            if any(len(self.tr_data[field][k]) == 0 for k in self.tr_data[field]):
                self.load_time_resolved(field)
            if not hasattr(self, 'avg_tr_data') or field not in self.avg_tr_data:
                self.avg_time_resolved(field)
            if not hasattr(self, 'avg_tr_collection_data') or field not in self.avg_tr_collection_data:
                self.avg_time_resolved_collections(field)

        elif source == 'in':
            if not self.in_bool:
                raise ValueError('Interval data not found')
            if field not in self.in_fields:
                raise ValueError(f'Field {field!r} not in interval fields: {self.in_fields}')
            if any(np.array_equal(self.in_data[field][c][0], 0) for c in self.in_data[field]):
                self.load_intervals(field)
            if not hasattr(self, 'avg_time_avg_in_data') or field not in self.avg_time_avg_in_data:
                self.avg_intervals_over_time(field)
            if not hasattr(self, 'time_avg_in_data') or field not in self.time_avg_in_data:
                self.avg_intervals_by_collection_over_time(field)

        else:
            raise ValueError(f'source must be one of "ta", "tr", "in"; got {source!r}')

    def _color_chooser(self, idx, num_colors, cmap='GnBu', reverse=False):
        '''
        Choose a color from a list of colors

        Parameters
        ----------
        idx : int
            The index of the color to choose
        num_colors : int
            The number of colors in the list
        cmap : str, default='GnBu'
            The colormap to use
        reverse : bool, default=False
            Whether to reverse the colormap

        Returns
        -------
        str
            The color
        '''
        cmap = plt.get_cmap(cmap)
        if reverse:
            cmap = cmap.reversed()
        return cmap((idx + 1)/ (num_colors + 1))

    def _get_x_data_and_label(self, x_length, field, edf_type):
        """
        Get the x-axis data and label for a given field.

        Parameters
        ----------
        x_length : int
            The length of the x-axis data to be returned.
        field : str
            The field name.
        edf_type : str, optional
            The EDF type if the field is an EDF field.

        Returns
        -------
        np.ndarray
            The x-axis data.
        str
            The x-axis label.
        """
        EDF_PREFIXES = ('EDF', 'ExDF', 'EyDF', 'EzDF')
        if x_length == len(self.cells) and not field.startswith(EDF_PREFIXES):
            x = self.cells
            xlabel = 'Position [m]'
        elif x_length == len(self.nodes) and not field.startswith(EDF_PREFIXES):
            x = self.nodes
            xlabel = 'Position [m]'
        elif field.startswith(EDF_PREFIXES):
            x = self.edf_energy[edf_type]
            xlabel = 'Energy [eV]'
        else:
            raise ValueError('Could not get x-axis data')

        return x, xlabel

    def get_time_averaged_power(self, type: str, species: str):
        '''
        Calculate the power deposited into a species from either capacitive
        or inductive heating.

        Parameters
        ----------
        type : str
            The type of power to calculate. Must be either
            'capacitive' ('c', 'cap', 'P_C') or 'inductive' ('i', 'ind', 'P_I')

        species : str
            The species name to calculate the power for.

        Returns
        -------
        float
            The total power deposited into the species over the entire simulation
        np.ndarray
            The power deposited into the species as a function of
            diagnostic collection
        '''
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if type.lower() in ['capacitive', 'c', 'cap', 'p_c']:
            field = f'P_C_{species}'
        elif type.lower() in ['inductive', 'i', 'ind', 'p_i']:
            field = f'P_I_{species}'
        else:
            raise ValueError('Type must be either "capacitive" or "inductive"')

        if field not in self.ta_data:
            try:
                self.avg_time_averaged(field)
            except Exception as e:
                raise ValueError(f'Field {field} not found in time averaged data')

        dz = self.dz

        power_by_coll = np.zeros(len(self.ta_data[field]), dtype=float)
        for coll in range(len(self.ta_data[field])):
            power_by_coll[coll] = np.sum(self.ta_data[field][coll + 1]) * dz
        total_power = np.average(power_by_coll)

        return total_power, power_by_coll

    def get_total_time_averaged_power(self, types: list, species_list: list):
        '''
        Calculate the total power deposited into a list of species from both
        capacitive or inductive heating.

        Parameters
        ----------
        types : list of str
            The types of power to calculate. Must be either
            'capacitive' ('c', 'cap', 'P_C') or 'inductive' ('i', 'ind', 'P_I')

        species_list : list
            A list of species names to calculate the power for.

        Returns
        -------
        float
            The total power deposited into the species over the entire simulation
        dict
            A dictionary with species names as keys and total power as values
        '''
        total_power = 0.0
        power_by_type = {}
        for type in types:
            power_by_type[type] = {}
            for species in species_list:
                species_power, _ = self.get_time_averaged_power(type, species)
                if species in power_by_type[type]:
                    power_by_type[type][species] += species_power
                else:
                    power_by_type[type][species] = species_power
                total_power += species_power

        return total_power, power_by_type
