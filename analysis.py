from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import os

class Analysis:
    def __init__(self, directory: str = './diags'):
        '''
        Initialize the Analysis object with the directory of the diagnostics data
        
        Parameters
        ----------
        directory : str
            The directory of the diagnostics data
        '''
        # Get the absolute path of the directory
        self.directory = os.path.abspath(directory)
        self.dir = os.listdir(directory)
        self.dir.sort()

        self.ieadf_bool = False
        self.Riz_bool = False
        self.in_bool = False
        self.tr_bool = False
        self.time_averaged_bool = False

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
            print('IEADF data found')
            self.ieadf_bool = True

            # Save the ieadf directories (there will be one for each ion species)
            temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('ieadf')]
            temp.sort()
            self.ieadf_dir = {}
            for species_dir in temp:
                # Save the species name as the dictionary key and the directory as the value
                self.ieadf_dir[species_dir.split('ieadf_')[-1]] = species_dir
            if len(self.ieadf_dir) > 1:
                print(f' - {len(self.ieadf_dir)} IEADF directories found for species: {", ".join(self.ieadf_dir.keys())}')
            else:
                print(f' - {len(self.ieadf_dir)} IEADF directory found for species: {", ".join(self.ieadf_dir.keys())}')

            # Initialize the energy and degree bin dictionaries
            self.energy = {}
            self.energy_edges = {}
            self.deg = {}
            self.deg_edges = {}

            # Initialize the left and right wall ieadf collection dictionaries
            self.lw_ieadf_colls = {}
            self.rw_ieadf_colls = {}

            # Initialize the ieadf data dictionary
            self.ieadf_data_lists = {}

            # Check if the ieadf directory has bins_eV.npy and bins_deg.npy
            for key, directory in self.ieadf_dir.items():
                print(f' - Looking into directory for species: {key}')
                ieadf_dir = os.listdir(directory)
                ieadf_dir.sort()
                if 'bins_eV.npy' in ieadf_dir:
                    self.energy[key] = np.load(directory + '/bins_eV.npy')
                    # Energies are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.energy_edges[key] = np.zeros(self.energy[key].size + 1)
                    self.energy_edges[key][0] = self.energy[key][0] - (self.energy[key][1] - self.energy[key][0])/2
                    self.energy_edges[key][1:-1] = (self.energy[key][1:] + self.energy[key][:-1])/2
                    self.energy_edges[key][-1] = self.energy[key][-1] + (self.energy[key][-1] - self.energy[key][-2])/2
                else:
                    print(f'   > Energy bins not found')
                if 'bins_deg.npy' in ieadf_dir:
                    self.deg[key] = np.load(directory + '/bins_deg.npy')
                    # Degrees are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.deg_edges[key] = np.zeros(self.deg[key].size + 1)
                    self.deg_edges[key][0] = self.deg[key][0] - (self.deg[key][1] - self.deg[key][0])/2
                    self.deg_edges[key][1:-1] = (self.deg[key][1:] + self.deg[key][:-1])/2
                    self.deg_edges[key][-1] = self.deg[key][-1] + (self.deg[key][-1] - self.deg[key][-2])/2
                else:
                    print(f'   > Degree bins not found')

                self.ieadf_data_lists[key] = {}
                # Print collected idfs
                if any(file.startswith('lw') for file in ieadf_dir):
                    self.lw_ieadf_colls[key] = [f'{directory}/{file}' for file in ieadf_dir if file.startswith('lw')]
                    self.lw_ieadf_colls[key].sort()
                    print(f'   > {len(self.lw_ieadf_colls[key])} left wall collections')
                    # Initialize the left wall ieadf data dictionary
                    self.ieadf_data_lists[key]['lw'] = []

                if any(file.startswith('rw') for file in ieadf_dir):
                    self.rw_ieadf_colls[key] = [f'{directory}/{file}' for file in ieadf_dir if file.startswith('rw')]
                    self.rw_ieadf_colls[key].sort()
                    print(f'   > {len(self.rw_ieadf_colls[key])} right wall collections')
                    # Initialize the right wall ieadf data dictionary
                    self.ieadf_data_lists[key]['rw'] = []

        # Check if any of the elements in self.dir start with r_ioniz
        if any(dir.startswith('r_ioniz') for dir in self.dir):
            print('Ionization rate data found')
            self.Riz_bool = True

            # Save the r_ioniz directories (there will be one for each ion species)
            temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('r_ioniz')]
            temp.sort()
            self.Riz_dir = {}
            for species_dir in temp:
                # Save the species name as the dictionary key and the directory as the value
                self.Riz_dir[species_dir.split('r_ioniz_')[-1]] = species_dir
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
                else:
                    print(f'   > Position bins not found')
                if 'bins_t.npy' in Riz_dir:
                    self.Riz_t[key] = np.load(directory + '/bins_t.npy')
                    # Times are cell midpoints, and we need to get the edges for plotting with plt.pcolormesh
                    self.Riz_t_edges[key]       = np.zeros(self.Riz_t[key].size + 1)
                    self.Riz_t_edges[key][0]    = self.Riz_t[key][0] - (self.Riz_t[key][1] - self.Riz_t[key][0])/2
                    self.Riz_t_edges[key][1:-1] = (self.Riz_t[key][1:] + self.Riz_t[key][:-1])/2
                    self.Riz_t_edges[key][-1]   = self.Riz_t[key][-1] + (self.Riz_t[key][-1] - self.Riz_t[key][-2])/2
                else:
                    print(f'   > Time bins not found')

                self.Riz_data_lists[key] = {}
                # Print collected data
                if any(file.startswith('Riz') for file in Riz_dir):
                    self.Riz_colls[key] = [f'{directory}/{file}' for file in Riz_dir if file.startswith('Riz')]
                    self.Riz_colls[key].sort()
                    print(f'   > {len(self.Riz_colls[key])} data collections')
                    # Initialize the data dictionary
                    self.Riz_data_lists[key] = []

        if any(dir.startswith('interval') for dir in self.dir):
            print('Interval data found')
            self.in_bool = True
            temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('interval')]
            temp.sort()
            self.in_colls = {}
            for coll in temp:
                self.in_colls[int(coll.split('/')[-1].split('_')[-1])] = coll
            num_colls = len(self.in_colls)
            if num_colls == 0:
                print(f' - {num_colls} interval collections found')
            else:
                # Open file f'{self.directory}/diagnostic_times.dat' to get the time intervals
                with open(f'{self.directory}/diagnostic_times.dat', 'r') as f:
                    for line in f:
                        if line.startswith('Times in interval='):
                            self.in_times = np.array([float(time) for time in line.split('=')[1].split(', ')])
                            break
                print(f' - {num_colls} interval collections at {len(self.in_times)} time intervals: {", ".join([str(time) for time in self.in_times])}')

                # Print collected fields
                self.in_fields = [file.split('.')[0] for file in os.listdir(self.in_colls[1]) if file.endswith('.npz')]
                self.in_fields.sort()
                print(f' - {len(self.in_fields)} fields: {", ".join(self.in_fields)}')

                # Set up dictionary to store interval data
                self.in_data = {}
                for field in self.in_fields:
                    self.in_data[field] = {}
                    for collection in self.in_colls:
                        self.in_data[field][collection] = [0]*len(self.in_times)

        if any(dir.startswith('time_resolved') for dir in self.dir):
            print('Time resolved data found')
            self.tr_bool = True
            temp = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('time_resolved')]
            temp.sort()
            self.tr_colls = {}
            for coll in temp:
                self.tr_colls[int(coll.split('/')[-1].split('_')[-1])] = coll
            num_colls = len(self.tr_colls)
            print(f' - {num_colls} time resolved collections')

            if num_colls > 0:
                # Print collected fields
                self.tr_fields = [file.split('.')[0] for file in os.listdir(self.tr_colls[1]) if file.endswith('.npy') and file != 'times.npy']
                self.tr_fields.sort()
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

        if any(dir.startswith('time_averaged') for dir in self.dir):
            print('Time averaged data found')
            self.time_averaged_bool = True
            self.time_averaged_colls = [f'{self.directory}/{dir}' for dir in self.dir if dir.startswith('time_averaged')]
            self.time_averaged_colls.sort()
            print(f' - {len(self.time_averaged_colls)} time averaged collections')

            # Print collected fields
            self.time_averaged_fields = [file.split('.')[0] for file in os.listdir(self.time_averaged_colls[0]) if file.endswith('.npy')]
            self.time_averaged_fields.sort()
            print(f' - {len(self.time_averaged_fields)} fields: {", ".join(self.time_averaged_fields)}')

            # Set up dictionary to store time averaged data
            self.time_averaged_data = {}
            for field in self.time_averaged_fields:
                self.time_averaged_data[field] = {}
                for collection in self.time_averaged_colls:
                    self.time_averaged_data[field][int(collection.split('/')[-1].split('_')[-1])] = []

        if self.in_bool or self.tr_bool or self.time_averaged_bool:
            self.cells = np.load(f'{self.directory}/cells.npy')
            self.nodes = np.load(f'{self.directory}/nodes.npy')

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
            ax.plot(self.energy[species], Riz[species], label = species)
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
                      normalize: bool = True,
                      dpi=150):
        '''
        Plot the collection-averaged IEDF data

        Parameters
        ----------
        species : str, default=None
            The species to plot. If None, plots all species on a single axis
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
            self.get_avg_iedf_data()
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
                        ax.plot(self.energy[spec], iedfs[spec][wall], label = wall)
                    ax.set_ylim(0, np.max([np.max(iedfs[spec][wall]) for wall in iedfs[spec]])*1.05)
                    ax.legend()
                else:
                    ax.plot(self.energy[spec], iedfs[spec])
                    ax.set_ylim(0, np.max(iedfs[spec])*1.05)
            ax.set_xlabel('Energy [eV]')
            ax.set_ylabel('IEDF [eV$^{-1}$]')
            ax.set_title('Simulation IEDF')
            ax.margins(x=0)
        else:
            fig, ax = plt.subplots(1,1, dpi=dpi)
            if isinstance(iedfs[species], dict):
                for wall in iedfs[species]:
                    ax.plot(self.energy[species], iedfs[species][wall], label = wall)
                ax.set_ylim(0, np.max([np.max(iedfs[species][wall]) for wall in iedfs[species]])*1.05)
                ax.legend()
            else:
                ax.plot(self.energy[species], iedfs[species])
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
                    self.normalized_iedfs[species][wall] = self.avg_iedf_data[species][wall] / np.trapz(self.avg_iedf_data[species][wall], self.energy[species])
            else:
                self.normalized_iedfs[species] = self.avg_iedf_data[species] / np.trapz(self.avg_iedf_data[species], self.energy[species])

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
                cbar = ax.pcolormesh(self.deg_edges[spec], self.energy_edges[spec], ieadfs[spec], shading='auto')
                fig.colorbar(cbar, ax=ax, label='IEADF [eV$^{-1}$]')
                ax.set_xlabel('Degrees')
                ax.set_ylabel('Energy [eV]')
                ax.set_title(f'{spec} IEADF')
            return figs, axs
        else:
            if isinstance(ieadfs[species], dict):
                raise NotImplementedError('Cannot plot ieadfs with separate left and right wall data yet. Needs to be implemented.')
            fig, ax = plt.subplots(1,1, dpi=dpi)
            cbar = ax.pcolormesh(self.deg_edges[species], self.energy_edges[species], ieadfs[species], shading='auto')
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
            area_factor = np.abs(np.sin(self.deg[species] * np.pi / 180))
            area_factor = np.tile(area_factor, (self.energy[species].size, 1)) # Resize area factor to be size (energy.size, deg.size)
            for ii in range(len(self.energy[species])):
                area_factor[ii] = np.sqrt(self.energy[species][ii]) * area_factor[ii] # Multiply each row by the corresponding energy bin to caluclate the area factor

            # Check if the species have been separated into left and right wall data
            if isinstance(self.avg_ieadf_data[species], dict):
                self.normalized_ieadfs[species] = {}
                for wall in self.avg_ieadf_data[species]:
                    self.normalized_ieadfs[species][wall] = self.avg_ieadf_data[species][wall] / np.trapz(np.trapz(self.avg_ieadf_data[species][wall], self.energy[species], axis=0), self.deg[species]) / area_factor
            else:
                self.normalized_ieadfs[species] = self.avg_ieadf_data[species] / np.trapz(np.trapz(self.avg_ieadf_data[species], self.energy[species], axis=0), self.deg[species]) / area_factor

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
    
    def plot_avg_interval(self, field: str, interval: int = None, dpi : int = 150, cmap : str = 'GnBu'):
        '''
        Plot the average interval data
        
        Parameters
        ----------
        field : str
            The field to plot
        interval : int, default=None
            The index (from 0 to len(self.interval_times - 1)) of the interval
            to plot. If None, plots all intervals on a single axis.
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
        fig, ax = plt.subplots(1,1, dpi=dpi)
        if field == 'E_z' or field == 'J_d':
            x = self.cells
        else:
            x = self.nodes
        if interval is None:
            num = len(self.in_times)
            for ii in range(num):
                ax.plot(x, self.avg_in_data[field][ii],
                        label = f't={self.in_times[ii]}*T',
                        color = self._color_chooser(ii, num, cmap = cmap))
            ax.set_title(f'{field} intervals')

            # Make avg line
            if not hasattr(self, 'avg_tr_data'):
                self.avg_time_resolved(field)
            if field not in self.avg_tr_data:
                self.avg_time_resolved(field)
            # Plot avg line
            ax.plot(x, self.avg_tr_data[field], label = 'Average', color = 'black')
            ax.legend(loc = [1.01,0], fontsize = 'small')

        else:
            ax.plot(x, self.avg_in_data[field][interval])
            ax.set_title(f'{field} at t = {self.in_times[interval]}*T')
        ax.set_xlabel('Position [m]')
        ax.set_ylabel(f'{field}')
        ax.margins(x=0)

        return fig, ax

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

    def plot_avg_time_resolved_collection(self, field: str, collection: int = None, dpi = 150, cmap : str = 'GnBu'):
        '''
        Plot the average time resolved data
        
        Parameters
        ----------
        field : str
            The field to plot
        collection : int, default=None
            The index of the collection to plot. If None, plots all collections
            on a single axis
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
        fig, ax = plt.subplots(1,1, dpi=dpi)
        if field == 'E_z' or field == 'J_d':
            x = self.cells
        else:
            x = self.nodes
        if collection is None:
            # Plot lines from each collection
            num = len(self.avg_tr_collection_data[field])
            for coll in self.avg_tr_collection_data[field]:
                ax.plot(x, self.avg_tr_collection_data[field][coll],
                        label = f't={self.tr_times[coll][len(self.tr_times[coll]) // 2]:.4e}',
                        color = self._color_chooser(coll, num, cmap=cmap))
                
            # Make avg line
            if not hasattr(self, 'avg_tr_data'):
                self.avg_time_resolved(field)
            if field not in self.avg_tr_data:
                self.avg_time_resolved(field)
            # Plot avg line
            ax.plot(x, self.avg_tr_data[field], label = 'Average', color = 'black')

        else:
            ax.plot(x, self.avg_tr_collection_data[field][collection],
                    label = f't={self.tr_times[collection][len(self.tr_times[collection]) // 2]:.4e}')

        ax.set_xlabel('Position [m]')
        ax.set_ylabel(f'{field}')
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)
        ax.legend(loc = [1.01,0], fontsize = 'small')

        return fig, ax
    
    def plot_avg_time_resolved(self, field: str, dpi=150):
        '''
        Plot the average time resolved data
        
        Parameters
        ----------
        field : str
            The field to plot
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
        if not hasattr(self, 'avg_tr_data'):
            self.avg_time_resolved(field)
        if field not in self.avg_tr_data:
            self.avg_time_resolved(field)
        fig, ax = plt.subplots(1,1, dpi=dpi)
        if field == 'E_z' or field == 'J_d':
            x = self.cells
        else:
            x = self.nodes
        ax.plot(x, self.avg_tr_data[field], label='Average', color = 'black')
        ax.set_xlabel('Position [m]')
        ax.set_ylabel(f'{field}')
        ax.set_title(f'Time averaged {field}')
        ax.margins(x=0)

        return fig, ax

    def animate_time_resolved(self, field: str, collection: int, dpi=150, interval=100, repeat_delay=500):
        '''
        Animate the time resolved data
        
        Parameters
        ----------
        field : str
            The field to animate
        collection : int
            The index of the collection to animate
        dpi : int
            The DPI of the plot
        interval : int, default=200
            The interval between frames in milliseconds
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
        # Check if the field has been loaded into self.tr_data. If it unloaded, the list will be empty
        if any([len(self.tr_data[field][key]) == 0 for key in self.tr_data[field]]):
            self.load_time_resolved(field)

        fig, ax = plt.subplots(1,1, dpi=dpi)
        if field == 'E_z' or field == 'J_d':
            x = self.cells
        else:
            x = self.nodes
        line, = ax.plot(x, self.tr_data[field][collection][0], color='black')

        ax.set_xlabel('Position [m]')
        ax.set_ylabel(f'{field}')
        ax.set_title(f'Time resolved {field}')
        ax.margins(x=0)

        def update(frame):
            line.set_ydata(self.tr_data[field][collection][frame])
            min = np.min(self.tr_data[field][collection][frame])
            max = np.max(self.tr_data[field][collection][frame])
            
            if max == min:
                ax.set_ylim(max - 0.01*max, max + 0.01*max)
            else:
                ax.set_ylim(min, max)
            return line,

        anim = FuncAnimation(fig, update, frames=len(self.tr_data[field][collection]), interval=interval, repeat_delay=repeat_delay)
        return anim
    
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