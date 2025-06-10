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
        # Set the list of diagnostics that plot on the cells
        self.cell_diags = ['E_z', 'J_d', 'CPe', 'CPi', 'IPe', 'IPi']

        # Get the absolute path of the directory
        self.directory = os.path.abspath(directory)
        self.dir = os.listdir(directory)
        self.dir.sort()

        self.ieadf_bool = False
        self.Riz_bool = False
        self.in_bool = False
        self.tr_bool = False
        self.ta_bool = False

        # Open file f'{self.directory}/diagnostic_times.dat' to get the cell size and time step
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

        # Check if any of the elements in self.dir start with ieadf
        if any(dir.startswith('ieadf') for dir in self.dir):
            if not quiet_startup:
                print('IEADF data found')
            self.ieadf_bool = True

            # Save the ieadf directories (there will be one for each ion species)
            temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('ieadf')]
            temp.sort()
            self.ieadf_dir = {}
            for species_dir in temp:
                # Save the species name as the dictionary key and the directory as the value
                self.ieadf_dir[species_dir.split('ieadf_')[-1]] = species_dir
            if not quiet_startup:
                if len(self.ieadf_dir) > 1:
                    print(f' - {len(self.ieadf_dir)} IEADF directories found for species: {", ".join(self.ieadf_dir.keys())}')
                else:
                    print(f' - {len(self.ieadf_dir)} IEADF directory found for species: {", ".join(self.ieadf_dir.keys())}')

            # Initialize the energy and degree bin dictionaries
            self.ieadf_energy = {}
            self.ieadf_energy_edges = {}
            self.ieadf_deg = {}
            self.ieadf_deg_edges = {}

            # Initialize the left and right wall ieadf collection dictionaries
            self.lw_ieadf_colls = {}
            self.rw_ieadf_colls = {}

            # Initialize the ieadf data dictionary
            self.ieadf_data_lists = {}

            # Check if the ieadf directory has bins_eV.npy and bins_deg.npy
            for key, directory in self.ieadf_dir.items():
                if not quiet_startup:
                    print(f' - Looking into directory for species: {key}')
                ieadf_dir = os.listdir(directory)
                ieadf_dir.sort()
                if 'bins_eV.npy' in ieadf_dir:
                    self.ieadf_energy[key] = np.load(directory + '/bins_eV.npy')
                    # Energies are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.ieadf_energy_edges[key] = np.zeros(self.ieadf_energy[key].size + 1)
                    self.ieadf_energy_edges[key][0] = self.ieadf_energy[key][0] - (self.ieadf_energy[key][1] - self.ieadf_energy[key][0])/2
                    self.ieadf_energy_edges[key][1:-1] = (self.ieadf_energy[key][1:] + self.ieadf_energy[key][:-1])/2
                    self.ieadf_energy_edges[key][-1] = self.ieadf_energy[key][-1] + (self.ieadf_energy[key][-1] - self.ieadf_energy[key][-2])/2
                elif not quiet_startup:
                    print(f'   > Energy bins not found')
                if 'bins_deg.npy' in ieadf_dir:
                    self.ieadf_deg[key] = np.load(directory + '/bins_deg.npy')
                    # Degrees are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.ieadf_deg_edges[key] = np.zeros(self.ieadf_deg[key].size + 1)
                    self.ieadf_deg_edges[key][0] = self.ieadf_deg[key][0] - (self.ieadf_deg[key][1] - self.ieadf_deg[key][0])/2
                    self.ieadf_deg_edges[key][1:-1] = (self.ieadf_deg[key][1:] + self.ieadf_deg[key][:-1])/2
                    self.ieadf_deg_edges[key][-1] = self.ieadf_deg[key][-1] + (self.ieadf_deg[key][-1] - self.ieadf_deg[key][-2])/2
                elif not quiet_startup:
                    print(f'   > Degree bins not found')

                self.ieadf_data_lists[key] = {}
                # Print collected idfs
                if any(file.startswith('lw') for file in ieadf_dir):
                    self.lw_ieadf_colls[key] = [f'{directory}/{file}' for file in ieadf_dir if file.startswith('lw')]
                    self.lw_ieadf_colls[key].sort()
                    if not quiet_startup:
                        print(f'   > {len(self.lw_ieadf_colls[key])} left wall collections')
                    # Initialize the left wall ieadf data dictionary
                    self.ieadf_data_lists[key]['lw'] = []

                if any(file.startswith('rw') for file in ieadf_dir):
                    self.rw_ieadf_colls[key] = [f'{directory}/{file}' for file in ieadf_dir if file.startswith('rw')]
                    self.rw_ieadf_colls[key].sort()
                    if not quiet_startup:
                        print(f'   > {len(self.rw_ieadf_colls[key])} right wall collections')
                    # Initialize the right wall ieadf data dictionary
                    self.ieadf_data_lists[key]['rw'] = []

        # Check if any of the elements in self.dir start with r_ioniz
        if any(dir.startswith('r_ioniz') for dir in self.dir):
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

            # Initialize the z and time bin dictionaries
            self.Riz_z = {}
            self.Riz_z_edges = {}
            self.Riz_t = {}
            self.Riz_t_edges = {}

            # Initialize the collection dictionary
            self.Riz_colls = {}

            # Initialize the data dictionary
            self.Riz_data_lists = {}

            # Check if the ieadf directory has bins_z.npy and bins_t.npy
            for key, directory in self.Riz_dir.items():
                if not quiet_startup:
                    print(f' - Looking into directory for species: {key}')
                Riz_dir = os.listdir(directory)
                Riz_dir.sort()
                if 'bins_z.npy' in Riz_dir:
                    self.Riz_z[key] = np.load(directory + '/bins_z.npy')
                    # Positions are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.Riz_z_edges[key]       = np.zeros(self.Riz_z[key].size + 1)
                    self.Riz_z_edges[key][0]    = self.Riz_z[key][0] - (self.Riz_z[key][1] - self.Riz_z[key][0])/2
                    self.Riz_z_edges[key][1:-1] = (self.Riz_z[key][1:] + self.Riz_z[key][:-1])/2
                    self.Riz_z_edges[key][-1]   = self.Riz_z[key][-1] + (self.Riz_z[key][-1] - self.Riz_z[key][-2])/2
                elif not quiet_startup:
                    print(f'   > Position bins not found')
                if 'bins_t.npy' in Riz_dir:
                    self.Riz_t[key] = np.load(directory + '/bins_t.npy')
                    # Times are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.Riz_t_edges[key]       = np.zeros(self.Riz_t[key].size + 1)
                    self.Riz_t_edges[key][0]    = self.Riz_t[key][0] - (self.Riz_t[key][1] - self.Riz_t[key][0])/2
                    self.Riz_t_edges[key][1:-1] = (self.Riz_t[key][1:] + self.Riz_t[key][:-1])/2
                    self.Riz_t_edges[key][-1]   = self.Riz_t[key][-1] + (self.Riz_t[key][-1] - self.Riz_t[key][-2])/2
                elif not quiet_startup:
                    print(f'   > Time bins not found')

                self.Riz_data_lists[key] = {}
                # Print collected data
                if any(file.startswith('Riz') for file in Riz_dir):
                    self.Riz_colls[key] = [f'{directory}/{file}' for file in Riz_dir if file.startswith('Riz')]
                    self.Riz_colls[key].sort()
                    if not quiet_startup:
                        print(f'   > {len(self.Riz_colls[key])} data collections')
                    # Initialize the data dictionary
                    self.Riz_data_lists[key] = []

        if any(dir.startswith('interval') for dir in self.dir):
            if not quiet_startup:
                print('Interval data found')
            self.in_bool = True
            temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('interval')]
            temp.sort()
            self.in_colls = {}
            for coll in temp:
                self.in_colls[int(coll.split('/')[-1].split('_')[-1])] = coll
            num_colls = len(self.in_colls)
            if num_colls == 0 and not quiet_startup:
                print(f' - {num_colls} interval collections found')
            else:
                # Open file f'{self.directory}/diagnostic_times.dat' to get the time intervals
                with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
                    for line in f:
                        if line.startswith('Times in interval='):
                            self.in_times = np.array([float(time) for time in line.split('=')[1].split(', ')])
                            break
                if not quiet_startup:
                    print(f' - {num_colls} interval collections at {len(self.in_times)} time intervals: {", ".join([str(time) for time in self.in_times])}')

                # Print collected fields
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

        if any(dir.startswith('time_resolved') for dir in self.dir):
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

            if num_colls > 0:
                # Print collected fields
                self.tr_fields = [file.split('.')[0] for file in os.listdir(self.tr_colls[1]) if file.endswith('.npy') and file != 'times.npy']
                self.tr_fields.sort()
                if not quiet_startup:
                    print(f' - {len(self.tr_fields)} fields: {", ".join(self.tr_fields)}')

                # Set up dictionary to store time resolved data
                self.tr_data = {}
                for field in self.tr_fields:
                    self.tr_data[field] = {}
                    for collection in self.tr_colls:
                        self.tr_data[field][collection] = []
                # Set up dictionary to store times of resolved collection
                self.tr_times = {}
                for collection in self.tr_data[field]:
                    self.tr_times[collection] = np.load(f'{self.tr_colls[collection]}/times.npy')

            # Get the interval time in case avg_time_resolved_over_collections is called
            with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
                for line in f:
                    if line.startswith('Interval period [s]='):
                        self.interval_period = float(line.split('=')[1])
                        break
            if not quiet_startup:
                print(f' - Assuming an RF period of {self.interval_period:.2e} s')

        if any(dir.startswith('time_averaged') for dir in self.dir):
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

            # Print collected fields
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

        if self.in_bool or self.tr_bool or self.ta_bool:
            self.cells = np.load(f'{self.directory}/cells.npy')
            self.nodes = np.load(f'{self.directory}/nodes.npy')

        # Collect energy distribution function bins for normal edfs, if they exist
        edf_fields = ['EEdf', 'IEdf']
        if any(hasattr(self, attr) and any(field.startswith(edf) for field in getattr(self, attr)) for attr in ['ta_fields', 'tr_fields', 'in_fields'] for edf in edf_fields):
            if not quiet_startup:
                print('Energy distribution function data found')
            # Get the boundaries of the edf collection region from the diagnostic_times.dat file
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

            # Determine indices for all boundaries including domain edges
            self.edf_cell_indices = np.r_[0, np.searchsorted(self.nodes, self.edf_box_boundaries, side='left'), len(self.nodes)-1]
            # Append a boundary at zero and at the end of the domain
            self.edf_box_boundaries = np.concatenate(([0], self.edf_box_boundaries, [self.nodes[-1]]))
            if not quiet_startup:
                print(f' - Edfs collected in {len(self.edf_box_boundaries) - 1} regions')
            self.edf_box_midpoints = (self.edf_box_boundaries[:-1] + self.edf_box_boundaries[1:]) / 2

            self.edf_energy = {}
            for edf in edf_fields:
                if any(hasattr(self, attr) and any(field.startswith(edf) for field in getattr(self, attr)) for attr in ['ta_fields', 'tr_fields', 'in_fields']):
                    self.edf_energy[edf] = np.load(f'{self.directory}/{edf.lower()}_bins_eV.npy')
                    if not quiet_startup:
                        print(f' - {edf} energy bins collected')

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
        # Load the ieadf data
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
            ax.plot(self.ieadf_energy[species], Riz[species], label = species)
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

    def load_ieadf_data_lists(self, species: str = None):
        '''
        Load the IEADF data

        Parameters
        ----------
        species : str, default=None
            The species to load

        Returns
        -------
        ieadf_data_lists : dict[dict[list[np.ndarray]]]
            The IEADF data organized like
            ieadf_data_lists[species][wall][collection]
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        if species is not None:
            if species not in self.ieadf_dir:
                raise ValueError(f'Species must be one of: {", ".join(self.ieadf_dir.keys())}')
            # Add left wall data, if necessary
            if 'lw' in self.ieadf_data_lists[species]:
                self.ieadf_data_lists[species]['lw'] = [np.load(coll) for coll in self.lw_ieadf_colls[species]]
            # Add right wall data, if necessary
            if 'rw' in self.ieadf_data_lists[species]:
                self.ieadf_data_lists[species]['rw'] = [np.load(coll) for coll in self.rw_ieadf_colls[species]]
        else:
            for spec in self.ieadf_dir:
                if 'lw' in self.ieadf_data_lists[spec]:
                    self.ieadf_data_lists[spec]['lw'] = [np.load(coll) for coll in self.lw_ieadf_colls[spec]]
                if 'rw' in self.ieadf_data_lists[spec]:
                    self.ieadf_data_lists[spec]['rw'] = [np.load(coll) for coll in self.rw_ieadf_colls[spec]]

        return self.ieadf_data_lists

    def get_avg_ieadf_data(self, separate_rl: bool = False):
        '''
        Get the average IEADF over all collections. Optionally, average the
        left and right wall data separately.

        Parameters
        ----------
        separate_rl : bool, default=False
            Average the left and right wall IEADF data separately if True

        Returns
        -------
        avg_ieadf_data : dict[np.ndarray] or dict[dict[np.ndarray]]
            The averaged IEADF data for each species. If separate_rl is False,
            the data is organized like avg_ieadf_data[species]. If separate_rl
            is True, the data is organized like avg_ieadf_data[species][wall].
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        # Load the ieadf data
        self.load_ieadf_data_lists()

        # Average both walls data together
        if not separate_rl:
            # Initialize the dictionary to store the average IEADF data
            self.avg_ieadf_data = {}
            for species in self.ieadf_data_lists:
                temp_array_list = []
                for wall in self.ieadf_data_lists[species]:
                    for array in self.ieadf_data_lists[species][wall]:
                        temp_array_list.append(array)
                # Save the average IEADF data for the species
                self.avg_ieadf_data[species] = np.mean(temp_array_list, axis=0)
        else:
            # Initialize the dictionary to store the average IEADF data
            self.avg_ieadf_data = {}
            for species in self.ieadf_data_lists:
                self.avg_ieadf_data[species] = {}
                for wall in self.ieadf_data_lists[species]:
                    temp_array_list = []
                    for array in self.ieadf_data_lists[species][wall]:
                        temp_array_list.append(array)
                    self.avg_ieadf_data[species][wall] = np.mean(temp_array_list, axis=0)

        return self.avg_ieadf_data

    def get_iedf_data_lists(self):
        '''
        Gets IEDF data from the list of IEADF data.

        Returns
        -------
        iedf_data_lists : dict[dict[list[np.ndarray]]]
            The IEDF data
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        self.load_ieadf_data_lists()
        self.iedf_data_lists = {}
        for species in self.ieadf_data_lists:
            self.iedf_data_lists[species] = {}
            for wall in self.ieadf_data_lists[species]:
                self.iedf_data_lists[species][wall] = []
                for array in self.ieadf_data_lists[species][wall]:
                    self.iedf_data_lists[species][wall].append(np.sum(array, axis=1))
        return self.iedf_data_lists

    def get_avg_iedf_data(self, separate_rl: bool = False):
        '''
        Get the average IEDF over all collections. Optionally, average the
        left and right wall data separately.

        Parameters
        ----------
        separate_rl : bool, default=False
            Average the left and right wall IEDF data separately if True

        Returns
        -------
        avg_iedf_data : dict[np.ndarray] or dict[dict[np.ndarray]]
            The IEDF data for each species. If separate_rl is False, the data
            is organized like avg_iedf_data[species]. If separate_rl is True,
            the data is organized like avg_iedf_data[species][wall].
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        if not hasattr(self, 'iedf_data_lists'):
            self.get_iedf_data_lists()

        # Average both walls data together
        if not separate_rl:
            # Initialize the dictionary to store the average IEDF data
            self.avg_iedf_data = {}
            for species in self.iedf_data_lists:
                temp_array_list = []
                for wall in self.iedf_data_lists[species]:
                    for array in self.iedf_data_lists[species][wall]:
                        temp_array_list.append(array)
                # Save the average IEDF data for the species
                self.avg_iedf_data[species] = np.mean(temp_array_list, axis=0)
        else:
            # Initialize the dictionary to store the average IEDF data
            self.avg_iedf_data = {}
            for species in self.iedf_data_lists:
                self.avg_iedf_data[species] = {}
                for wall in self.iedf_data_lists[species]:
                    temp_array_list = []
                    for array in self.iedf_data_lists[species][wall]:
                        temp_array_list.append(array)
                    self.avg_iedf_data[species][wall] = np.mean(temp_array_list, axis=0)

        return self.avg_iedf_data

    def plot_avg_iedf(self,
                      species: str = None,
                      separate_rl: bool = False,
                      normalize: bool = True,
                      dpi=150):
        '''
        Plot the collection-averaged IEDF data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on a single axis
        separate_rl : bool, default=False
            Average the left and right wall IEDF data separately if True
        normalize : bool
            Normalize the IEDF data
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        if not hasattr(self, 'avg_iedf_data'):
            self.get_avg_iedf_data(separate_rl=separate_rl)
        if species is not None and species not in self.iedf_data_lists:
            raise ValueError(f'Species must be one of: {", ".join(self.iedf_data_lists.keys())}')
        if normalize:
            iedfs = self.normalize_iedf()
        else:
            iedfs = self.avg_iedf_data
        if species is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            for spec in iedfs:
                if isinstance(iedfs[spec], dict):
                    for wall in iedfs[spec]:
                        ax.plot(self.ieadf_energy[spec], iedfs[spec][wall], label = wall)
                    ax.set_ylim(0, np.max([np.max(iedfs[spec][wall]) for wall in iedfs[spec]])*1.05)
                    ax.legend()
                else:
                    ax.plot(self.ieadf_energy[spec], iedfs[spec])
                    ax.set_ylim(0, np.max(iedfs[spec])*1.05)
            ax.set_xlabel('Energy [eV]')
            ax.set_ylabel('IEDF [eV$^{-1}$]')
            ax.set_title('Simulation IEDF')
            ax.margins(x=0)
        else:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            if isinstance(iedfs[species], dict):
                for wall in iedfs[species]:
                    ax.plot(self.ieadf_energy[species], iedfs[species][wall], label = wall)
                ax.set_ylim(0, np.max([np.max(iedfs[species][wall]) for wall in iedfs[species]])*1.05)
                ax.legend()
            else:
                ax.plot(self.ieadf_energy[species], iedfs[species])
                ax.set_ylim(0, np.max(iedfs[species])*1.05)
            ax.set_xlabel('Energy [eV]')
            ax.set_ylabel('IEDF [eV$^{-1}$]')
            ax.set_title('Simulation IEDF')
            ax.margins(x=0)

        return fig, ax

    def normalize_iedf(self):
        '''
        Normalize the collection-averaged IEDF data

        Returns
        -------
        iedf : dict[np.ndarray] or dict[dict[np.ndarray]]
            The normalized IEDF data, organized like iedf[species] or
            iedf[species][wall] based on how the data is organized coming in
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        if not hasattr(self, 'avg_iedf_data'):
            self.get_avg_iedf_data()
        self.normalized_iedfs = {}
        for species in self.avg_iedf_data:
            # Check if the species have been separated into left and right wall data
            if isinstance(self.avg_iedf_data[species], dict):
                self.normalized_iedfs[species] = {}
                for wall in self.avg_iedf_data[species]:
                    integral = np.trapezoid(self.avg_iedf_data[species][wall], self.ieadf_energy[species])
                    if integral > 0:
                        self.normalized_iedfs[species][wall] = self.avg_iedf_data[species][wall] / integral
                    else:
                        self.normalized_iedfs[species][wall] = np.zeros_like(self.avg_iedf_data[species][wall])
            else:
                integral = np.trapezoid(self.avg_iedf_data[species], self.ieadf_energy[species])
                if integral > 0:
                    self.normalized_iedfs[species] = self.avg_iedf_data[species] / integral
                else:
                    self.normalized_iedfs[species] = np.zeros_like(self.avg_iedf_data[species])

        return self.normalized_iedfs

    def plot_avg_ieadf(self,
                       species: str = None,
                       normalize: bool = True,
                       dpi=150):
        '''
        Plot the collection-averaged IEADF data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on a separate figs
        normalize : bool
            Normalize the IEADF data
        dpi : int
            The DPI of the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        if not hasattr(self, 'avg_ieadf_data'):
            self.get_avg_ieadf_data()
        if species is not None and species not in self.avg_ieadf_data:
            raise ValueError(f'Species must be one of: {", ".join(self.avg_ieadf_data.keys())}')
        if normalize:
            ieadfs = self.normalize_ieadf()
        else:
            ieadfs = self.avg_ieadf_data
        if species is None:
            figs = []
            axs = []
            for spec in ieadfs:
                if isinstance(ieadfs[spec], dict):
                    raise NotImplementedError('Cannot plot ieadfs with separate left and right wall data yet. Needs to be implemented.')
                fig, ax = plt.subplots(1,1, dpi=dpi)
                figs.append(fig)
                axs.append(ax)
                cbar = ax.pcolormesh(self.ieadf_deg_edges[spec], self.ieadf_energy_edges[spec], ieadfs[spec], shading='auto')
                fig.colorbar(cbar, ax=ax, label='IEADF [eV$^{-1}$]')
                ax.set_xlabel('Degrees')
                ax.set_ylabel('Energy [eV]')
                ax.set_title(f'{spec} IEADF')
            return figs, axs
        else:
            if isinstance(ieadfs[species], dict):
                raise NotImplementedError('Cannot plot ieadfs with separate left and right wall data yet. Needs to be implemented.')
            fig, ax = plt.subplots(1,1, dpi=dpi)
            cbar = ax.pcolormesh(self.ieadf_deg_edges[species], self.ieadf_energy_edges[species], ieadfs[species], shading='auto')
            fig.colorbar(cbar, ax=ax, label='IEADF [eV$^{-1}$]')
            ax.set_xlabel('Degrees')
            ax.set_ylabel('Energy [eV]')
            ax.set_title(f'{species} IEADF')
            return fig, ax

    def normalize_ieadf(self):
        '''
        Normalize the collection-averaged IEADF data

        Returns
        -------
        ieadf : dict[np.ndarray]
            The normalized IEADF data
        '''
        if not self.ieadf_bool:
            raise ValueError('IEADF data not found')
        if not hasattr(self, 'avg_ieadf_data'):
            self.get_avg_ieadf_data()
        self.normalized_ieadfs = {}
        for species in self.avg_ieadf_data:

            # Get the area factor to normalize the IEADF data. To use, divide by the area factor.
            # Area factor is the sine of the angle multiplied by the square root of the energy
            area_factor = np.abs(np.sin(self.ieadf_deg[species] * np.pi / 180))
            area_factor = np.tile(area_factor, (self.ieadf_energy[species].size, 1)) # Resize area factor to be size (energy.size, deg.size)
            for ii in range(len(self.ieadf_energy[species])):
                area_factor[ii] = np.sqrt(self.ieadf_energy[species][ii]) * area_factor[ii] # Multiply each row by the corresponding energy bin to caluclate the area factor

            # Check if the species have been separated into left and right wall data
            if isinstance(self.avg_ieadf_data[species], dict):
                self.normalized_ieadfs[species] = {}
                for wall in self.avg_ieadf_data[species]:
                    self.normalized_ieadfs[species][wall] = self.avg_ieadf_data[species][wall] / np.trapz(np.trapz(self.avg_ieadf_data[species][wall], self.ieadf_energy[species], axis=0), self.ieadf_deg[species]) / area_factor
            else:
                self.normalized_ieadfs[species] = self.avg_ieadf_data[species] / np.trapz(np.trapz(self.avg_ieadf_data[species], self.ieadf_energy[species], axis=0), self.ieadf_deg[species]) / area_factor

        return self.normalized_ieadfs

    def load_intervals(self, field: str = None):
        '''
        Load the interval data

        Parameters
        ----------
        field : str
            The field to load

        Returns
        -------
        in_data : dict[dict[list[np.ndArray]]]
            The interval data
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        if field is not None:
            if field not in self.in_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.in_fields)}')

            for coll in self.in_data[field]:
                temp = np.load(f'{self.in_colls[coll]}/{field}.npz')
                # Unpack elements of the npz file from t01 to t{self.interval_times.size+1} into elements of a list
                for ii in range(len(self.in_times)):
                    self.in_data[field][coll][ii] = temp[f't{ii+1:02d}']
        else:
            for fld in self.in_fields:
                for coll in self.in_data[fld]:
                    temp = np.load(f'{self.in_colls[coll]}/{fld}.npz')
                    # Unpack elements of the npz file from t01 to t{self.interval_times.size+1} into elements of a list
                    for ii in range(len(self.in_times)):
                        self.in_data[fld][coll][ii] = temp[f't{ii+1:02d}']
        return self.in_data

    def add_interval_field(self, field: str):
        '''
        Add an interval field to the interval data

        Parameters
        ----------
        field : str
            The field to add. Must be one of 'P_e', 'P_i', 'P_t', 'EfV',
            'Jzc'

        Returns
        -------
        in_data[field] : dict[stack of np.ndArray]
            The interval data
        '''
        if not self.in_bool:
            raise ValueError('Interval data not found')
        if field not in ['P_e', 'P_i', 'P_t', 'EfV', 'Jzc']:
            raise ValueError('Field must be one of: P_e, P_i, P_t, EfV, Jzc')
        if field in ['P_e', 'P_i', 'P_t']:
            if field not in self.in_fields:
                self.add_interval_field('EfV')
            if field == 'P_e':
                self.load_intervals('Jze')
                self.in_data[field] = {}
                for coll in self.in_data['EfV']:
                    self.in_data[field][coll] = [0] * len(self.in_times)
                    for interval in range(len(self.in_times)):
                        self.in_data[field][coll][interval] = self.in_data['EfV'][coll][interval] * self.in_data['Jze'][coll][interval]
                # Check if the field is already in self.in_fields before adding
                if field not in self.in_fields:
                    self.in_fields.append(field)
            elif field == 'P_i':
                self.load_intervals('Jzi')
                self.in_data[field] = {}
                for coll in self.in_data['EfV']:
                    self.in_data[field][coll] = [0] * len(self.in_times)
                    for interval in range(len(self.in_times)):
                        self.in_data[field][coll][interval] = self.in_data['EfV'][coll][interval] * self.in_data['Jzi'][coll][interval]
                # Check if the field is already in self.in_fields before adding
                if field not in self.in_fields:
                    self.in_fields.append(field)
            elif field == 'P_t':
                self.load_intervals('Jze')
                self.load_intervals('Jzi')
                self.in_data[field] = {}
                for coll in self.in_data['EfV']:
                    self.in_data[field][coll] = [0] * len(self.in_times)
                    for interval in range(len(self.in_times)):
                        self.in_data[field][coll][interval] = self.in_data['EfV'][coll][interval] * (self.in_data['Jze'][coll][interval] + self.in_data['Jzi'][coll][interval])
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
        if not self.in_bool:
            raise ValueError('Interval data not found')
        if not hasattr(self, 'avg_in_data'):
            self.avg_intervals(field)
        if field not in self.avg_in_data:
            self.avg_intervals(field)

        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            return_fig = True

        # Make avg line
        if plot_time_avg:
            if not hasattr(self, 'avg_tr_data'):
                self.avg_time_resolved(field)
            if field not in self.avg_tr_data:
                self.avg_time_resolved(field)

        # Get x-axis data
        if len(self.avg_in_data[field][0]) == len(self.cells):
            x = self.cells
            xlabel = 'Position [m]'
        elif len(self.avg_in_data[field][0]) == len(self.nodes):
            x = self.nodes
            xlabel = 'Position [m]'
        elif field.startswith('EEdf'):
            x = self.edf_energy['EEdf']
            xlabel = 'Energy [eV]'
        elif field.startswith('IEdf'):
            x = self.edf_energy['IEdf']
            xlabel = 'Energy [eV]'
        else:
            raise ValueError('Could not get x-axis data')

        if interval is None:
            num = len(self.in_times)
            for ii in range(num):
                ax.plot(x, self.avg_in_data[field][ii],
                        label = f't={self.in_times[ii]}*T',
                        color = self._color_chooser(ii, num, cmap = cmap))
            ax.set_title(f'{field} intervals')

            # Plot avg line
            if plot_time_avg:
                ax.plot(x, self.avg_tr_data[field], label = 'Average', color = 'black')
            ax.legend(loc = [1.01,0], fontsize = 'small')

        else:
            ax.plot(x, self.avg_in_data[field][interval],
                    label = f't={self.in_times[interval]}*T',
                    color = self._color_chooser(interval, len(self.in_times), cmap = cmap))
            ax.set_title(f'{field} at t = {self.in_times[interval]}*T')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'{field}')
        ax.margins(x=0)

        if return_fig:
            return fig, ax
        else:
            return ax

    def load_time_resolved(self, field: str = None):
        '''
        Load the time resolved data

        Parameters
        ----------
        field : str
            The field to load

        Returns
        -------
        tr_data : dict[dict[stack of np.ndArray]]
            The time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field is not None:
            if field not in self.tr_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.tr_fields)}')
            for coll in self.tr_data[field]:
                self.tr_data[field][coll] = np.load(f'{self.tr_colls[coll]}/{field}.npy')
        else:
            for fld in self.tr_fields:
                for coll in self.tr_data[fld]:
                    self.tr_data[fld][coll] = np.load(f'{self.tr_colls[coll]}/{fld}.npy')
        return self.tr_data

    def add_time_resolved_field(self, field: str):
        '''
        Add a time resolved field to the time resolved data

        Parameters
        ----------
        field : str
            The field to add. Must be one of 'P_e', 'P_i', 'P_t', 'EfV',
            'Jzc'

        Returns
        -------
        tr_data : dict[dict[stack of np.ndArray]]
            The time resolved data
        '''
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if field not in ['P_e', 'P_i', 'P_t', 'EfV', 'Jzc']:
            raise ValueError('Field must be one of: P_e, P_i, P_t, EfV, Jzc')
        if field in ['P_e', 'P_i', 'P_t']:
            if field not in self.tr_fields:
                self.add_time_resolved_field('EfV')
            if field == 'P_e':
                self.load_time_resolved('Jze')
                self.tr_data[field] = {}
                for coll in self.tr_data['EfV']:
                    self.tr_data[field][coll] = self.tr_data['EfV'][coll] * self.tr_data['Jze'][coll]
                # Check if the field is already in self.tr_fields before adding
                if field not in self.tr_fields:
                    self.tr_fields.append(field)
            elif field == 'P_i':
                self.load_time_resolved('Jzi')
                self.tr_data[field] = {}
                for coll in self.tr_data['EfV']:
                    self.tr_data[field][coll] = self.tr_data['EfV'][coll] * self.tr_data['Jzi'][coll]
                # Check if the field is already in self.tr_fields before adding
                if field not in self.tr_fields:
                    self.tr_fields.append(field)
            elif field == 'P_t':
                self.load_time_resolved('Jze')
                self.load_time_resolved('Jzi')
                self.tr_data[field] = {}
                for coll in self.tr_data['EfV']:
                    self.tr_data[field][coll] = self.tr_data['EfV'][coll] * (self.tr_data['Jze'][coll] + self.tr_data['Jzi'][coll])
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
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')
        if not hasattr(self, 'avg_tr_collection_data'):
            self.avg_time_resolved_collections(field)
        if field not in self.avg_tr_collection_data:
            self.avg_time_resolved_collections(field)

        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            return_fig = True

        # Make avg line
        if not hasattr(self, 'avg_tr_data'):
            self.avg_time_resolved(field)
        if field not in self.avg_tr_data:
            self.avg_time_resolved(field)

        # Get x-axis data
        if len(self.avg_tr_data[field]) == len(self.cells):
            x = self.cells
            xlabel = 'Position [m]'
        elif len(self.avg_tr_data[field]) == len(self.nodes):
            x = self.nodes
            xlabel = 'Position [m]'
        elif field.startswith('EEdf'):
            x = self.edf_energy['EEdf']
            xlabel = 'Energy [eV]'
        elif field.startswith('IEdf'):
            x = self.edf_energy['IEdf']
            xlabel = 'Energy [eV]'
        else:
            raise ValueError('Could not get x-axis data')

        if collection is None:
            # Plot lines from each collection
            num = len(self.avg_tr_collection_data[field])
            for coll in self.avg_tr_collection_data[field]:
                ax.plot(x, self.avg_tr_collection_data[field][coll],
                        label = f't={self.tr_times[coll][len(self.tr_times[coll]) // 2]:.4e}',
                        color = self._color_chooser(coll, num, cmap=cmap))

            # Plot avg line
            ax.plot(x, self.avg_tr_data[field], label = 'Average', color = 'black')

        else:
            ax.plot(x, self.avg_tr_collection_data[field][collection],
                    label = f't={self.tr_times[collection][len(self.tr_times[collection]) // 2]:.4e}')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'{field}')
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)
        ax.legend(loc = [1.01,0], fontsize = 'small')

        if return_fig:
            return fig, ax
        else:
            return ax

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
        if not self.tr_bool:
            raise ValueError('Time resolved data not found')

        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            return_fig = True

        # Make avg line
        if not hasattr(self, 'avg_tr_data'):
            self.avg_time_resolved(field)
        if field not in self.avg_tr_data:
            self.avg_time_resolved(field)

        # Get x-axis data
        if len(self.avg_tr_data[field]) == len(self.cells):
            x = self.cells
            xlabel = 'Position [m]'
        elif len(self.avg_tr_data[field]) == len(self.nodes):
            x = self.nodes
            xlabel = 'Position [m]'
        elif field.startswith('EEdf'):
            x = self.edf_energy['EEdf']
            xlabel = 'Energy [eV]'
        elif field.startswith('IEdf'):
            x = self.edf_energy['IEdf']
            xlabel = 'Energy [eV]'
        else:
            raise ValueError('Could not get x-axis data')

        ax.plot(x, self.avg_tr_data[field], label='Average', color = 'black')
        ax.set_xlabel('Position [m]')
        ax.set_ylabel(f'{field}')
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)

        if return_fig:
            return fig, ax
        else:
            return ax

    def animate_time_resolved(self,
                              field: str,
                              collection: int = None,
                              title: str = None,
                              xlabel: str = None,
                              ylabel: str = None,
                              color: str = None,
                              xlim: list[tuple] = None,
                              ylim: list[tuple] = None,
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
        if len(data[0]) == len(self.cells):
            x = self.cells
            if set_xlabel_flag:
                xlabel = 'Position [m]'
        elif len(data[0]) == len(self.nodes):
            x = self.nodes
            if set_xlabel_flag:
                xlabel = 'Position [m]'
        elif field.startswith('EEdf'):
            x = self.edf_energy['EEdf']
            if set_xlabel_flag:
                xlabel = 'Energy [eV]'
        elif field.startswith('IEdf'):
            x = self.edf_energy['IEdf']
            if set_xlabel_flag:
                xlabel = 'Energy [eV]'
        else:
            raise ValueError('Could not get x-axis data')

        # Plot initial frame
        line, = ax.plot(x, data[0], color=color)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x=0)

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

        # Set default matplotlib style
        plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': ticklabelsize, 'ytick.labelsize': ticklabelsize})

        # If 2 field, make them next to each other. If more than 2, make a grid (2x2, 3x3, etc.)
        if num_plots == 1:
            fig, axs = plt.subplots(1,1, dpi=dpi)
        elif num_plots == 2:
            fig, axs = plt.subplots(1,2, dpi=dpi)
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
            if len(data[ii][0]) == len(self.cells):
                x.append(self.cells)
                if set_xlabel_flag:
                    xlabel.append('Position [m]')
            elif len(data[ii][0]) == len(self.nodes):
                x.append(self.nodes)
                if set_xlabel_flag:
                    xlabel.append('Position [m]')
            elif fld.startswith('EEdf'):
                x.append(self.edf_energy['EEdf'])
                if set_xlabel_flag:
                    xlabel.append('Energy [eV]')
            elif fld.startswith('IEdf'):
                x.append(self.edf_energy['IEdf'])
                if set_xlabel_flag:
                    xlabel.append('Energy [eV]')
            else:
                raise ValueError(f'Could not get x-axis data for {fld}')

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

        # Set default matplotlib style
        plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': ticklabelsize, 'ytick.labelsize': ticklabelsize})

        # If 2 field, make them next to each other. If more than 2, make a grid (2x2, 3x3, etc.)
        if num_plots == 1:
            fig, axs = plt.subplots(1,1, dpi=dpi)
        elif num_plots == 2:
            fig, axs = plt.subplots(1,2, dpi=dpi)
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
            if len(data[ii][0]) == len(self.cells):
                x.append(self.cells)
                if set_xlabel_flag:
                    xlabel.append('Position [m]')
            elif len(data[ii][0]) == len(self.nodes):
                x.append(self.nodes)
                if set_xlabel_flag:
                    xlabel.append('Position [m]')
            elif fld.startswith('EEdf'):
                x.append(self.edf_energy['EEdf'])
                if set_xlabel_flag:
                    xlabel.append('Energy [eV]')
            elif fld.startswith('IEdf'):
                x.append(self.edf_energy['IEdf'])
                if set_xlabel_flag:
                    xlabel.append('Energy [eV]')
            else:
                raise ValueError(f'Could not get x-axis data for {fld}')

        # Plot initial frame
        lines = []
        for ii, ax in enumerate(axs.flat):
            if ii >= len(field):
                ax.axis('off')
                continue

            if field[ii].startswith('EEdf'):
                norm_data = data[ii][0] / (np.sum(data[ii][0] * np.diff(self.edf_energy['EEdf'])[0]) * self.edf_energy['EEdf'] ** (0.5))
                tmp_line, = ax.plot(x[ii], norm_data, color=color[ii])
                ax.set_yscale('log')
            else:
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
                if field[ii].startswith('EEdf'):
                    norm_data = data[ii][frame] / (np.sum(data[ii][frame] * np.diff(self.edf_energy['EEdf'])[0]) * self.edf_energy['EEdf'] ** (0.5))
                    line.set_ydata(norm_data)
                else:
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
            The field to load

        Returns
        -------
        ta_data : dict[dict[stack of np.ndArray]]
            The time averaged data
        '''
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if field is not None:
            if field not in self.ta_fields:
                raise ValueError(f'Field must be one of: {", ".join(self.ta_fields)}')
            for coll in self.ta_data[field]:
                self.ta_data[field][coll] = np.load(f'{self.ta_colls[coll]}/{field}.npy')
        else:
            for fld in self.ta_fields:
                for coll in self.ta_data[fld]:
                    self.ta_data[fld][coll] = np.load(f'{self.ta_colls[coll]}/{fld}.npy')
        return self.ta_data

    def add_time_averaged_field(self, field: str):
        '''
        Add a time averaged field to the time averaged data

        Parameters
        ----------
        field : str
            The field to add. Must be one of 'P_e', 'P_i', 'P_t', 'EfV',
            'Jzc'

        Returns
        -------
        tr_data : dict[dict[stack of np.ndArray]]
            The time resolved data
        '''
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if field not in ['P_e', 'P_i', 'P_t', 'EfV', 'Jzc']:
            raise ValueError('Field must be one of: P_e, P_i, P_t, EfV, Jzc')
        if field in ['P_e', 'P_i', 'P_t']:
            if field not in self.ta_fields:
                self.add_time_averaged_field('EfV')
            if field == 'P_e':
                self.load_time_averaged('Jze')
                self.ta_data[field] = {}
                for coll in self.ta_data['EfV']:
                    self.ta_data[field][coll] = self.ta_data['EfV'][coll] * self.ta_data['Jze'][coll]
                # Check if the field is already in self.ta_fields before adding
                if field not in self.ta_fields:
                    self.ta_fields.append(field)
            elif field == 'P_i':
                self.load_time_averaged('Jzi')
                self.ta_data[field] = {}
                for coll in self.ta_data['EfV']:
                    self.ta_data[field][coll] = self.ta_data['EfV'][coll] * self.ta_data['Jzi'][coll]
                # Check if the field is already in self.ta_fields before adding
                if field not in self.ta_fields:
                    self.ta_fields.append(field)
            elif field == 'P_t':
                self.load_time_averaged('Jze')
                self.load_time_averaged('Jzi')
                self.ta_data[field] = {}
                for coll in self.ta_data['EfV']:
                    self.ta_data[field][coll] = self.ta_data['EfV'][coll] * (self.ta_data['Jze'][coll] + self.ta_data['Jzi'][coll])
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
            if any([len(self.ta_data[field][key]) == 0 for key in self.ta_data[field]]):
                self.load_time_averaged(field)
            if not hasattr(self, 'avg_ta_data'):
                self.avg_ta_data = {}
            self.avg_ta_data[field] = np.mean([self.ta_data[field][coll] for coll in self.ta_data[field]], axis=0)
        else:
            self.avg_ta_data = {}
            for fld in self.ta_fields:
                if any([len(self.ta_data[fld][key]) == 0 for key in self.ta_data[fld]]):
                    self.load_time_averaged(fld)
                self.avg_ta_data[fld] = np.mean([self.ta_data[fld][coll] for coll in self.ta_data[fld]], axis=0)

    def plot_time_averaged(self,
                           field: str,
                           plot_all_coll = True,
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
        if not self.ta_bool:
            raise ValueError('Time averaged data not found')
        if field not in self.ta_fields:
            raise ValueError(f'Field must be one of: {", ".join(self.ta_fields)}')
        if edf_log_plot and not field.startswith('EEdf') and not field.startswith('IEdf'):
            raise ValueError('Field must be one of: EEdf, IEdf')
        # Check if the field has been loaded. If it unloaded, the list will be empty
        if any([len(self.ta_data[field][key]) == 0 for key in self.ta_data[field]]):
            self.load_time_averaged(field)

        if edf_log_plot:
            if field.startswith('EEdf'):
                edf_type = 'EEdf'
            elif field.startswith('IEdf'):
                edf_type = 'IEdf'

        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            return_fig = True

        # Make avg line
        if not hasattr(self, 'avg_ta_data'):
            self.avg_time_averaged(field)
        if field not in self.avg_ta_data:
            self.avg_time_averaged(field)

        # Get x-axis data
        if len(self.avg_ta_data[field]) == len(self.cells):
            x = self.cells
            xlabel = 'Position [m]'
        elif len(self.avg_ta_data[field]) == len(self.nodes):
            x = self.nodes
            xlabel = 'Position [m]'
        elif field.startswith('EEdf'):
            x = self.edf_energy['EEdf']
            xlabel = 'Energy [eV]'
        elif field.startswith('IEdf'):
            x = self.edf_energy['IEdf']
            xlabel = 'Energy [eV]'
        else:
            raise ValueError('Could not get x-axis data')

        if plot_all_coll:
            num = len(self.ta_data[field])
            for coll in self.ta_data[field]:
                data = self.ta_data[field][coll]
                if edf_log_plot:
                    data /= self.edf_energy[edf_type] ** (0.5)
                ax.plot(x, self.ta_data[field][coll],
                        label = f'Collection {coll}',
                        alpha = 0.4,
                        color = self._color_chooser(coll, num, cmap=cmap))

        if edf_log_plot:
            ax.plot(x, self.avg_ta_data[field] / self.edf_energy[edf_type] ** (0.5),
                    label='Average', color = 'black')
        else:
            ax.plot(x, self.avg_ta_data[field],
                    label='Average', color = 'black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'{field}')
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)
        if plot_all_coll:
            ax.legend(fontsize = 'small')
        if edf_log_plot:
            ax.set_yscale('log')

        if return_fig:
            return fig, ax
        else:
            return ax

    def _color_chooser(self, idx, num_colors, cmap='GnBu'):
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

        Returns
        -------
        str
            The color
        '''
        cmap = plt.get_cmap(cmap)
        return cmap((idx + 1)/ (num_colors + 1))