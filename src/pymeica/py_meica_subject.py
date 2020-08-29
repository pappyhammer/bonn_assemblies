from cicada.analysis.cicada_analysis_format_wrapper import CicadaAnalysisFormatWrapper
import os
import hdf5storage
import numpy as np
from sortedcontainers import SortedList, SortedDict
import neo
import quantities as pq
from pymeica.utils.spike_trains import create_spike_train_neo_format, spike_trains_threshold_by_firing_rate
import elephant.conversion as elephant_conv
from pymeica.utils.mcad import MCADOutcome
from pymeica.utils.file_utils import find_files
from pymeica.utils.misc import get_unit_label
import yaml
import pandas as pd


class SpikeStructure:

    def __init__(self, patient, spike_trains, microwire_labels, cluster_labels, cluster_indices,
                 spike_nums=None, title=None): # , ordered_indices=None, ordered_spike_data=None):
        """

        Args:
            patient:
            spike_nums:
            spike_trains:
            microwire_labels:
            cluster_labels:
            cluster_indices: indicate what is the index of the units among the clusters (indexing starts at 0)
            activity_threshold:
            title:
            ordered_indices:
            ordered_spike_data:
        """

        self.patient = patient
        self.spike_trains = spike_trains
        # self.ordered_spike_data = ordered_spike_data
        # array of int, representing the channel number actually such as in 'times_pos_CSC2.mat'
        self.microwire_labels = np.array(microwire_labels)
        # array of int
        self.cluster_labels = np.array(cluster_labels)
        self.cluster_indices = np.array(cluster_indices)
        self.title = title
        # cells labels
        self.labels = self.get_labels()
        # self.ordered_indices = ordered_indices
        # self.ordered_labels = None
        # self.ordered_spike_trains = None
        # if self.ordered_indices is not None:
        #     self.ordered_spike_trains = list()
        #     for index in ordered_indices:
        #         self.ordered_spike_trains.append(self.spike_trains[index])
        #     self.ordered_labels = []
        #     # y_ticks_labels_ordered = spike_nums_struct.labels[best_seq]
        #     for old_cell_index in self.ordered_indices:
        #         self.ordered_labels.append(self.labels[old_cell_index])

    def get_labels(self):
        labels = []
        # cluster_to_label = {1: "MU ", 2: "SU ", -1: "Artif ", 0: ""}
        cluster_to_label = {1: "MU", 2: "SU", -1: "Artif", 0: ""}
        # print(f"get_labels self.microwire_labels {self.microwire_labels}")
        for i, micro_wire in enumerate(self.microwire_labels):
            channel = self.patient.channel_info_by_microwire[micro_wire]
            unit_label = get_unit_label(cluster_label=cluster_to_label[self.cluster_labels[i]],
                                        cluster_index=self.cluster_indices[i],
                                        channel_index=micro_wire, region_label=channel)
            labels.append(unit_label)
            # labels.append(f"{cluster_to_label[self.cluster_labels[i]]}{micro_wire} "
            #               f"{channel}")
        return labels

    # def set_order(self, ordered_indices):
    #     if ordered_indices is None:
    #         self.ordered_spike_trains = np.copy(self.spike_trains)
    #     else:
    #         self.ordered_spike_trains = []
    #         for index in ordered_indices:
    #             self.ordered_spike_trains.append(self.spike_trains[index])
    #         self.ordered_indices = ordered_indices
    #         self.ordered_labels = []
    #         for old_cell_index in self.ordered_indices:
    #             self.ordered_labels.append(self.labels[old_cell_index])


class SleepStage:
    def __init__(self, number, start_time, stop_time, sleep_stage, conversion_datetime, conversion_timestamp):
        # timstamps are float, it's needed to multiple by 10^3 to get the real value, with represents microseconds
        self.start_time = start_time * 1000
        self.stop_time = stop_time * 1000
        # self.stop_time = stop_time * 1000
        # duration is in microseconds
        self.duration = self.stop_time - self.start_time
        self.duration_sec = self.duration / 1000000
        # string describing the stage (like "W", "3", "R")
        self.sleep_stage = sleep_stage
        self.conversion_datetime = conversion_datetime
        self.conversion_timestamp = conversion_timestamp * 1000
        self.number = number
        # first key is a tuple of int representing first_bin and last_bin
        # value is an instance of MCADOutcome, bin_size is available in MCADOutcome
        self.mcad_outcomes = SortedDict()
        # TODO: See to build an array with int key to get the MCADOutcome from a bin index or timestamps

    def add_mcad_outcome(self, mcad_outcome, bins_tuple):
        """
        Add an instance of MCADOutcome
        :param mcad_outcome:
        :param bins_tuple:
        :return:
        """
        self.mcad_outcomes[bins_tuple] = mcad_outcome

    def __str__(self):
        result = ""
        result += f"num  {self.number}, "
        result += f"stage  {self.sleep_stage}, "
        # result += f"start_time  {self.start_time}, "
        # result += f"stop_time  {self.stop_time}, \n"
        # result += f"duration (usec)  {self.duration}, "
        result += f"duration: {self.duration_sec:.1f} sec, {(self.duration / 1000000) / 60:.1f} min\n"
        if len(self.mcad_outcomes) == 0:
            result += f" No MCAD outcome\n"
        else:
            for bins_tuple, mcad_outcome in self.mcad_outcomes.items():
                first_bin_index = bins_tuple[0]
                last_bin_index = bins_tuple[1]
                chunk_duration = (last_bin_index - first_bin_index + 1) * mcad_outcome.spike_trains_bin_size
                # passing it in sec
                chunk_duration /= 1000
                result += f"{bins_tuple} {mcad_outcome.side} {mcad_outcome.n_cell_assemblies} cell " \
                          f"assemblies on {chunk_duration:.2f} sec segment.\n"
                if mcad_outcome.n_cell_assemblies > 0:
                    # cell_assembly is an instance of CellAssembly
                    for ca_index, cell_assembly in enumerate(mcad_outcome.cell_assemblies):
                        result += f"  CA n° {ca_index}: {cell_assembly.n_units} units, " \
                                  f"{cell_assembly.n_repeats} repeats, " \
                                  f"{cell_assembly.n_invariant_units} RU, " \
                                  f"{cell_assembly.n_responsive_units} IU, " \
                                  f"score: {cell_assembly.probability_score:.4f} \n"

        # result += f",\n conversion_datetime  {self.conversion_datetime}, "
        # result += f"conversion_timestamp  {self.conversion_timestamp}, "
        return result

#
"""
compatible with cicada.analysis.cicada_analysis_format_wrapper.CicadaAnalysisFormatWrapper
because it implements identifier decorator, as well as load_data and is_data_valid
But we don't import cicada code so this package could also be independent. 
"""
class PyMeicaSubject(CicadaAnalysisFormatWrapper):

    DATA_FORMAT = "PyMEICA"

    WRAPPER_ID = "PyMEICASubject"

    def __init__(self, data_ref, load_data=True, verbose=1):
        CicadaAnalysisFormatWrapper.__init__(self, data_ref=data_ref, data_format=self.DATA_FORMAT, load_data=load_data)

        # self._data_ref = data_ref
        # self.load_data_at_init = load_data
        # self._data_format = self.DATA_FORMAT

        # always load at the start
        self._identifier = os.path.basename(data_ref)
        self.verbose = verbose

        # variables initiated in self.load_data()
        # number of units, one Multi-unit count as one unit
        self.n_units = 0

        # The variable 'spikes' stores the spike shape from all spikes
        # measured in this channel.
        # This variable contains a matrix with dimension N_spikes x 64.
        # Each row corresponds to a single spike and gives 64 voltage values
        # of this spike aligned to the maximum.
        self.spikes_by_microwire = dict()

        # The variable 'cluster_class' provides information about the timing of
        # each spike and the cluster that it corresponds to.
        # This variable contains a N_spikes x 2 matrix in which the first
        # column contains the cluster that the spike belongs to and the
        # second column saves the time of the spike.
        # replace by the code of the type of unit: SU, MU etc... 1 = MU  2 = SU -1 = Artif.
        self.spikes_time_by_microwire = dict()

        self.channel_info_by_microwire = None

        # list of SleepStage instances (in chronological order)
        self.sleep_stages = list()

        self.cluster_info = None

        self.n_microwires = 0
        # self.available_micro_wires = 0

        self.nb_sleep_stages = 0
        # list of int, corresponding of the int representing the micro_wire such as in files 'times_CSC1'
        self.available_micro_wires = list()

        # key the stimulus number (int) and as value the string describing the
        #         stimulus (like "Barack Obama"). Init in load_stimuli_name
        self.stimuli_name_dict = dict()

        # key is the label (str) representing a unit such as 'MU 7 25 LMH2'
        # (SU or MU, cluster_index, microwireçindex, Side&Channel),
        # value is a list of two int representing the prefered stimulus in the evening and in the morning.
        # if -1, means no answer at this moment
        # if a label (cell) is not in this dict, it means it is not a responsive units
        self.is_responsive_units_dict = dict()
        # same as for is_responsive_units_dict but for invariant units
        self.is_invariant_units_dict = dict()

        if self.load_data_at_init:
            self.load_data()

    def _load_responsive_and_invariant_units(self, df, invariant_units):
        """

        :param df: panda dataframe to explore
        :param invariant_units: (bool) if True then it's invariant_units, else it is responsive units
        :return:
        """
        if invariant_units:
            units_dict = self.is_invariant_units_dict = dict()
        else:
            units_dict = self.is_responsive_units_dict = dict()

        df_response = df.loc[(df['Patient'] == int(self.identifier[1:3]))]
        if len(df_response) == 0:
            return
        # print(f"invariant_units {invariant_units}")
        for index in df_response.index:
            channel = df.loc[df.index[index], 'Channel']
            # removing one so it matches the actual indexing
            channel -= 1
            cluster = df.loc[df.index[index], 'Cluster']
            hemisphere = df.loc[df.index[index], 'Hemisphere']
            region = df.loc[df.index[index], 'Region']
            wire = df.loc[df.index[index], 'Wire']
            preferred_stim_num_e = df.loc[df.index[index], 'preferred_stim_num_e']
            preferred_stim_num_m = df.loc[df.index[index], 'preferred_stim_num_m']
            # print(f"channel {channel}, cluster {cluster}, hemisphere {hemisphere}, region {region}, "
            #       f"wire {wire}, preferred_stim_num_e {preferred_stim_num_e}, "
            #       f"preferred_stim_num_m {preferred_stim_num_m} ")
            # print(f"self.cluster_info[micro_wire] {len(self.cluster_info)}")

            cluster_infos = self.cluster_info[channel][0]
            cluster_match_index = cluster_infos[cluster]
            # print(f"cluster_infos {cluster_infos}")
            cluster_to_label = {1: "MU", 2: "SU", -1: "Artif", 0: ""}
            unit_label = get_unit_label(cluster_label=cluster_to_label[cluster_match_index],
                                        cluster_index=cluster,
                                        channel_index=channel,
                                        region_label=f"{hemisphere}{region}{wire}")
            units_dict[unit_label] = (preferred_stim_num_e, preferred_stim_num_m)

    @staticmethod
    def is_data_valid(data_ref):
        """
        Check if the data can be an input for this wrapper as data_ref
        Args:
            data_ref: file or directory

        Returns: a boolean

        """
        if not os.path.isdir(data_ref):
            return False

        files_in_dir = [item for item in os.listdir(data_ref)
                        if os.path.isfile(os.path.join(data_ref, item))]
        identifier = os.path.basename(data_ref)

        files_to_find = ["cluster_info.mat", f"{identifier}_sleepstages.mat"]
        for file_to_find in files_to_find:
            if file_to_find not in files_in_dir:
                return False

        return True

    # def get_sleep_stage_epoch(self, sleep_stage_name):
    #     """
    #     Return the epoch of a given type of slepe stage.
    #     :param sleep_stage_name:
    #     :return: List of list of 2 int represent the timestamps in sec of the beginning and end of each epoch
    #     """
    #     epochs = []
    #
    #     for sleep_stage in self.sleep_stages:
    #         if sleep_stage.sleep_stage != sleep_stage_name:
    #             continue
    #         epochs.append(sleep_stage.start_time, sleep_stage.stop_time)

    def load_data(self):
        CicadaAnalysisFormatWrapper.load_data(self)
        if self.verbose:
            print(f"Loading data for PyMeicaSubject {self._identifier}")
        # number of units, one Multi-unit count as one unit
        self.n_units = 0

        # Filter the items and only keep files (strip out directories)
        files_in_dir = [item for item in os.listdir(self._data_ref)
                        if os.path.isfile(os.path.join(self._data_ref, item))]

        # The variable 'spikes' stores the spike shape from all spikes
        # measured in this channel.
        # This variable contains a matrix with dimension N_spikes x 64.
        # Each row corresponds to a single spike and gives 64 voltage values
        # of this spike aligned to the maximum.
        self.spikes_by_microwire = dict()

        # The variable 'cluster_class' provides information about the timing of
        # each spike and the cluster that it corresponds to.
        # This variable contains a N_spikes x 2 matrix in which the first
        # column contains the cluster that the spike belongs to and the
        # second column saves the time of the spike.

        original_spikes_cluster_by_microwire = dict()

        # replace by the code of the type of unit: SU, MU etc... 1 = MU  2 = SU -1 = Artif.
        spikes_cluster_by_microwire = dict()
        self.spikes_time_by_microwire = dict()

        cluster_correspondance_by_microwire = dict()

        cluster_info_file = hdf5storage.loadmat(os.path.join(self._data_ref, "cluster_info.mat"))
        label_info = cluster_info_file["label_info"]
        # contains either an empty list if no cluster, or a list containing a list containing the type of cluster
        # 1 = MU  2 = SU -1 = Artif.
        # 0 = Unassigned (is ignored)
        self.cluster_info = cluster_info_file['cluster_info'][0, :]
        # adding cluster == 0 in index so it can match index in cluster_class
        for i, cluster in enumerate(self.cluster_info):
            if len(cluster) == 0:
                self.cluster_info[i] = [[0]]
            else:
                new_list = [0]
                new_list.extend(self.cluster_info[i][0])
                self.cluster_info[i] = [new_list]

        self.channel_info_by_microwire = cluster_info_file["cluster_info"][1, :]
        self.channel_info_by_microwire = [c[0] for c in self.channel_info_by_microwire]
        # print_mat_file_content(cluster_info_file)

        sleep_stages_file = hdf5storage.loadmat(os.path.join(self._data_ref, self._identifier + "_sleepstages.mat"),
                                                mat_dtype=True)
        conversion_datetime = sleep_stages_file["conversion_datetime"]
        conversion_timestamp = sleep_stages_file["conversion_timestamp"]

        # The variable 'sleepstages' is a N_sleepstages list size that contains 2 lists
        # with the first having 3 elements:
        # the starttime and stoptime of each sleep stage and the sleepstage label.
        sleep_stages_tmp = sleep_stages_file["sleepstages"][0, :]
        self.sleep_stages = []
        total_duration = 0
        for ss_index, sleep_stage_data in enumerate(sleep_stages_tmp):
            # sleep_stage_data = sleep_stage_data[0]
            # print(f"{ss_index} sleep_stage_data {sleep_stage_data}")
            # The start time of the first stage, might not be the same as the one of the first spike
            # recorded for this stage, as the data we have don't start at the beginning of a stage.
            ss = SleepStage(number=ss_index, start_time=sleep_stage_data[0][0][0], stop_time=sleep_stage_data[1][0][0],
                            sleep_stage=sleep_stage_data[2][0], conversion_datetime=conversion_datetime[0],
                            conversion_timestamp=conversion_timestamp[0][0])
            # print(f"ss {ss}")
            total_duration += ss.duration
            self.sleep_stages.append(ss)
        self.nb_sleep_stages = len(self.sleep_stages)
        # TODO: See to build a vector that give for any timestamps in the whole recording to which
        #  Sleepstage it belongs
        # print(f"sleepstages[0]: {sleepstages[1]}")
        # print(f"conversion_datetime {conversion_datetime}")
        # print(f"conversion_timestamp {conversion_timestamp[0][0]}")
        # print(f"conversion_timestamp int ? {isinstance(conversion_timestamp[0][0], int)}")
        if self.verbose:
            print(f"Data total duration (min): {(total_duration / 1000000) / 60}")
        # print_mat_file_content(sleep_stages_file)
        self.available_micro_wires = []
        for file_in_dir in files_in_dir:
            if file_in_dir.endswith("yaml") and (not file_in_dir.startswith(".")) and ("stimuli_name" in file_in_dir):
                self.load_stimuli_name(stimuli_yaml_file=os.path.join(self._data_ref, file_in_dir))
                continue

            # times_pos_CSC matched the full night recordings
            if file_in_dir.startswith("times_pos_CSC") or file_in_dir.startswith("times_CSC"):
                if file_in_dir.startswith("times_pos_CSC"):
                    # -1 to start by 0, to respect other matrices order
                    microwire_number = int(file_in_dir[13:-4]) - 1
                else:
                    microwire_number = int(file_in_dir[9:-4]) - 1
                self.available_micro_wires.append(microwire_number)
                data_file = hdf5storage.loadmat(os.path.join(self._data_ref, file_in_dir))
                # print(f"data_file {data_file}")
                self.spikes_by_microwire[microwire_number] = data_file['spikes']
                cluster_class = data_file['cluster_class']
                # .astype(int)
                # if value is 0, no cluster
                original_spikes_cluster_by_microwire = cluster_class[:, 0].astype(int)

                # spikes_cluster_by_microwire[microwire_number] = cluster_class[:, 0].astype(int)

                # changing the cluster reference by the cluster type, final values will be
                # 1 = MU  2 = SU -1 = Artif.
                # 0 = Unassigned (is ignored)
                # We want for each microwire to create as many lines of "units" as cluster
                go_for_debug_mode = False
                if go_for_debug_mode:
                    print(f"microwire_number {microwire_number}")
                    print(f"channel {self.channel_info_by_microwire[microwire_number]}")
                    print(f"self.cluster_info[microwire_number] {self.cluster_info[microwire_number]}")
                    print(f"np.unique(original_spikes_cluster_by_microwire) "
                          f"{np.unique(original_spikes_cluster_by_microwire)}")
                    print(f"original_spikes_cluster_by_microwire "
                          f"{original_spikes_cluster_by_microwire}")
                    print("")
                # for i, cluster_ref in enumerate(spikes_cluster_by_microwire[microwire_number]):
                #     if cluster_ref > 0:
                #         cluster_ref -= 1
                #         spikes_cluster_by_microwire[microwire_number][i] = \
                #             self.cluster_info[microwire_number][0][cluster_ref]

                # it's matlab indices, so we need to start with zero
                # not needed anymore because we add 0
                # for i, cluster_ref in enumerate(original_spikes_cluster_by_microwire):
                #     if cluster_ref > 0:
                #     original_spikes_cluster_by_microwire[i] -= 1

                # rounded to int
                # for each microwire, we add a dict with as many key as cluster, and for each key
                # we give as a value the spikes for this cluster
                self.spikes_time_by_microwire[microwire_number] = dict()

                # cluster_infos contains the list of clusters for this microwire.
                # original_spikes_cluster_by_microwire is same length as cluster_class[:, 1].astype(int), ie
                # nb spikes
                cluster_infos = self.cluster_info[microwire_number][0]
                for index_cluster, n_cluster in enumerate(cluster_infos):
                    # keep the spikes of the corresponding cluster
                    mask = np.zeros(len(cluster_class[:, 1]), dtype="bool")
                    mask[np.where(original_spikes_cluster_by_microwire == index_cluster)[0]] = True
                    # timstamps are float, it's needed to multiple by 10^3 to get the real value,
                    # represented as microseconds
                    self.spikes_time_by_microwire[microwire_number][index_cluster] = \
                        (cluster_class[mask, 1] * 1000)
                    # .astype(int)
                    # print(f"cluster_class[mask, 1] {cluster_class[mask, 1]}")
                    # print(f"- cluster_class[mask, 1] {cluster_class[mask, 1][0] - int(cluster_class[mask, 1][0])}")
                    self.n_units += 1

                if microwire_number < 0:
                    print(f"times_pos_CSC{microwire_number}")
                    print(f"spikes shape 0: {self.spikes_by_microwire[microwire_number][0, :]}")
                    # plt.plot(spikes_by_microwire[microwire_number][0, :])
                    # plt.show()
                    print(f"spikes cluster: {spikes_cluster_by_microwire[microwire_number]}")
                    # print(f"spikes time: {self.spikes_time_by_microwire[microwire_number].astype(int)}")
                    # print_mat_file_content(data_file)
                    print(f"\n \n")

        # 2nd round for responsive and invariant units
        for file_in_dir in files_in_dir:
            if file_in_dir.endswith("csv") and "responsive_units" in file_in_dir:
                # load responsive_units info
                responsive_units_file = os.path.join(self._data_ref, file_in_dir)
                responsive_units_df = pd.read_csv(responsive_units_file)
                self._load_responsive_and_invariant_units(df=responsive_units_df, invariant_units=False)
            elif file_in_dir.endswith("csv") and "invariant_units" in file_in_dir:
                # load invariant_units info
                invariant_units_file = os.path.join(self._data_ref, file_in_dir)
                invariant_units_df = pd.read_csv(invariant_units_file)
                self._load_responsive_and_invariant_units(df=invariant_units_df, invariant_units=True)

        self.n_microwires = len(self.spikes_by_microwire)
        self.available_micro_wires = np.array(self.available_micro_wires)

    def elapsed_time_from_falling_asleep(self, sleep_stage):
        """
        Looking at the time of the first sleep sleep_stage (it could start with Wake), return the number
        of time that separate it in seconds (could be negative if the stage is the wake one before sleep)
        :param sleep_stage: SleepStage instance
        :return:
        """
        for sleep_stage_index in range(len(self.sleep_stages)):
            ss = self.sleep_stages[sleep_stage_index]
            if ss.sleep_stage == "W":
                continue
            return (sleep_stage.start_time - ss.start_time) / 1000000
        return -1

    def load_stimuli_name(self, stimuli_yaml_file):
        """
        Load the file containing as key the stimulus number (int) and as value the string describing the
        stimulus (like "Barack Obama")
        :param stimuli_yaml_file:
        :return:
        """
        with open(stimuli_yaml_file, 'r') as stream:
            self.stimuli_name_dict = yaml.load(stream, Loader=yaml.Loader)

    def load_mcad_data(self, data_path, side_to_load=None,
                       sleep_stage_indices_to_load=None,
                       macd_comparison_key=MCADOutcome.BEST_SILHOUETTE,
                       min_repeat=3, update_progress_bar_fct=None,
                       time_started=None,
                       total_increment=1):
        """
        Explore all directories in data_path (recursively) and load the data issues from Malvache Cell Assemblies
        Detection code in yaml file.
        :param data_path:
        :param macd_comparison_key: indicate how to compare two outcomes for the same spike_trains section
        Choice among: MCADOutcome.BEST_SILHOUETTE & MCADOutcome.MAX_N_ASSEMBLIES
        :param min_repeat: minimum of times of cell assembly should repeat to be considered True.
        :param side_to_load: (str) if None, both side are loaded, otherwise should be 'L' or 'R'
        :param sleep_stage_indices_to_load: (list of int) if None, all stages are loaded, otherwise
        only the ones listed
        :param update_progress_bar_fct: for Cicada progress bar progress (optional), fct that take the initial time,
        and the increment at each step of the loading
        :return:
        """
        if data_path is None:
            return

        mcad_files = find_files(dir_to_explore=data_path, keywords=["stage"], extensions=("yaml", "yml"))

        # for progress bar purpose
        n_files = len(mcad_files)
        n_mcad_outcomes = 0
        increment_value = 0
        increment_step_for_files = (total_increment * 0.9) / n_files

        # first key: sleep_stage index, 2nd key: tuple of int representing firt and last bin,
        # value is a list of dict representing the content of the yaml file
        mcad_by_sleep_stage = dict()
        for file_index, mcad_file in enumerate(mcad_files):
            mcad_file_basename = os.path.basename(mcad_file)
            # to avoid loading the file, we filter based on the file_name, see to change if the file_names should
            # be changed, so far contain subject_id, sleep_index, side, bin of the chunk, stage_name
            if self.identifier not in mcad_file_basename:
                continue
            if (side_to_load is not None) and (side_to_load not in mcad_file_basename):
                continue
            if sleep_stage_indices_to_load is not None:
                split_file_name = mcad_file_basename.split()
                if split_file_name[4] == "index":
                    stage_index_from_file = int(mcad_file_basename.split()[5])
                else:
                    stage_index_from_file = int(mcad_file_basename.split()[3])
                if stage_index_from_file not in sleep_stage_indices_to_load:
                    continue

            with open(mcad_file, 'r') as stream:
                mcad_results_dict = yaml.load(stream, Loader=yaml.Loader)
                # first we check if it contains some of the field typical of mcad file
                if ("subject_id" not in mcad_results_dict) or ("sleep_stage_index" not in mcad_results_dict):
                    continue
                # then we check that it matches the actual subject_id
                if mcad_results_dict["subject_id"] != self.identifier:
                    continue
                sleep_stage_index = mcad_results_dict["sleep_stage_index"]
                first_bin_index = mcad_results_dict["first_bin_index"]
                last_bin_index = mcad_results_dict["last_bin_index"]
                bins_tuple = (first_bin_index, last_bin_index)
                side = mcad_results_dict["side"]
                if (side_to_load is not None) and (side != side_to_load):
                    continue
                if sleep_stage_indices_to_load is not None:
                    if sleep_stage_index not in sleep_stage_indices_to_load:
                        continue
                if sleep_stage_index not in mcad_by_sleep_stage:
                    mcad_by_sleep_stage[sleep_stage_index] = dict()
                if bins_tuple not in mcad_by_sleep_stage[sleep_stage_index]:
                    mcad_by_sleep_stage[sleep_stage_index][bins_tuple] = []
                mcad_by_sleep_stage[sleep_stage_index][bins_tuple].append(mcad_results_dict)
                n_mcad_outcomes += 1
            if update_progress_bar_fct is not None:
                increment_value += increment_step_for_files
                if increment_value > 1:
                    update_progress_bar_fct(time_started=time_started,
                                            increment_value=1)
                    increment_value -= 1

        # now we want to keep only one result for each chunk a given sleep_stage
        # and add it to the SleepStage instance

        increment_step_for_mcad_outcomes = (total_increment * 0.1) / n_mcad_outcomes

        for sleep_stage_index in mcad_by_sleep_stage.keys():
            for bins_tuple, mcad_dicts in mcad_by_sleep_stage[sleep_stage_index].items():
                best_mcad_outcome = None
                for mcad_dict in mcad_dicts:
                    mcad_outcome = MCADOutcome(mcad_yaml_dict=mcad_dict,
                                               comparison_key=macd_comparison_key,
                                               subject=self)
                    if update_progress_bar_fct is not None:
                        increment_value += increment_step_for_mcad_outcomes
                        if increment_value > 1:
                            update_progress_bar_fct(time_started=time_started,
                                                    increment_value=1)
                            increment_value -= 1

                    if best_mcad_outcome is None:
                        best_mcad_outcome = mcad_outcome
                    else:
                        best_mcad_outcome = best_mcad_outcome.best_mcad_outcome(mcad_outcome)

                if best_mcad_outcome.n_cell_assemblies == 0:
                    # if no cell assembly we don't keep it
                    continue

                # if one cell assembly we test that it repeats a minimum of time
                if best_mcad_outcome.n_cell_assemblies == 1:
                    if np.max(best_mcad_outcome.n_repeats_in_each_cell_assembly()) < min_repeat:
                        continue

                sleep_stage = self.sleep_stages[sleep_stage_index]

                sleep_stage.add_mcad_outcome(mcad_outcome=best_mcad_outcome,
                                             bins_tuple=best_mcad_outcome.bins_tuple)

    def build_spike_nums(self, sleep_stage_index, side_to_analyse, keeping_only_SU, remove_high_firing_cells,
                         firing_rate_threshold, spike_trains_binsize):
        """
        Build a spike_nums (bin version of spike_trains) from a sleep stage index and side.
        :param sleep_stage_index: (int)
        :param side_to_analyse: (str) 'L' or 'R'
        :param keeping_only_SU: (bool)
        :param remove_high_firing_cells: (bool)
        :param firing_rate_threshold: (int) in Hz
        :param spike_trains_binsize: (int) in ms
        :return:
        """

        spike_struct = self.construct_spike_structure(sleep_stage_indices=[sleep_stage_index],
                                                      channels_starting_by=[side_to_analyse],
                                                      keeping_only_SU=keeping_only_SU)
        spike_trains = spike_struct.spike_trains
        cells_label = spike_struct.labels
        binsize = spike_trains_binsize * pq.ms

        # first we create a spike_trains in the neo format
        spike_trains, t_start, t_stop = create_spike_train_neo_format(spike_trains)

        duration_in_sec = (t_stop - t_start) / 1000

        if remove_high_firing_cells:
            filtered_spike_trains, cells_below_threshold = \
                spike_trains_threshold_by_firing_rate(spike_trains=spike_trains,
                                                      firing_rate_threshold=firing_rate_threshold,
                                                      duration_in_sec=duration_in_sec)
            backup_spike_trains = spike_trains
            spike_trains = filtered_spike_trains
            n_cells_total = len(cells_label)
            cells_label_removed = [(index, label) for index, label in enumerate(cells_label) if
                                   index not in cells_below_threshold]
            cells_label = [label for index, label in enumerate(cells_label) if index in cells_below_threshold]
            n_cells = len(cells_label)
            # print(
            #     f"{n_cells_total - n_cells} cells had firing rate > {firing_rate_threshold} Hz and have been removed.")
            # if len(cells_label_removed):
            #     for index, label in cells_label_removed:
            #         print(f"{label}, {len(backup_spike_trains[index])}")

        n_cells = len(spike_trains)

        neo_spike_trains = []
        for cell in np.arange(n_cells):
            spike_train = spike_trains[cell]
            # print(f"n_spikes: {cells_label[cell]}: {len(spike_train)}")
            neo_spike_train = neo.SpikeTrain(times=spike_train, units='ms',
                                             t_start=t_start,
                                             t_stop=t_stop)
            neo_spike_trains.append(neo_spike_train)

        spike_trains_binned = elephant_conv.BinnedSpikeTrain(neo_spike_trains, binsize=binsize)

        # transform the binned spike train into array
        use_z_score_binned_spike_trains = False
        if use_z_score_binned_spike_trains:
            data = spike_trains_binned.to_array()
            # print(f"data.type() {type(data)}")
            # z-score
            spike_nums = np.zeros(data.shape, dtype="int8")
            for cell, binned_spike_train in enumerate(data):
                mean_train = np.mean(binned_spike_train)
                print(f"mean_train {mean_train} {np.max(binned_spike_train)}")
                binned_spike_train = binned_spike_train - mean_train
                n_before = len(np.where(data[cell] > 0)[0])
                n = len(np.where(binned_spike_train >= 0)[0])
                print(f"{cell}: n_before {n_before} vs {n}")
                spike_nums[cell, binned_spike_train >= 0] = 1
        else:
            spike_nums = spike_trains_binned.to_bool_array().astype("int8")

        return spike_trains, spike_nums, cells_label

    def construct_spike_structure(self, sleep_stage_indices=None,
                                  selection_range_time=None,
                                  sleep_stage_selection=None, channels_starting_by=None,
                                  channels_without_number=None, channels_with_number=None,
                                  title=None, keeping_only_SU=False):
        """
        Construct a spike structure (instance of SpikeStructure containing a spike trains with the labels
        corresponding to each spike train. There might be big gap between spike train in case two non contiguous
        time interval are included.

        :param selection_range_time: (tuple of float) represents two timestamps determining an epoch over which
        to select the spikes. If given, It is prioritized over sleep stages selection.
        :param sleep_stage_indices: (list of int) list of sleep_stage_indices
        :param sleep_stage_selection: (list of str) list of sleep stage according to their identifier. All sleep stages
        in this category will be added
        :param channels: list of str, if empty list, take them all, otherwise take the one starting with the same
        name (like "RAH" take RAH1, RAH2 etc...., if just "R" take all microwire on the right)
        :param channels_to_study: full name without numbers
        :param keeping_only_SU: if True, MU are also included
        :return:
        """
        # TODO: See to add in spike structure an option to know when there are time gaps

        # print(f"construct_spike_structure start for {self.identifier}")

        # don't put non-assigned clusters
        only_SU_and_MU = True

        micro_wire_to_keep = []
        if (channels_starting_by is None) and (channels_without_number is None) and (channels_with_number is None):
            micro_wire_to_keep = self.available_micro_wires
        else:
            if channels_starting_by is None:
                channels_starting_by = []

            if channels_without_number is None:
                channels_without_number = []

            if channels_with_number is None:
                channels_with_number = []
            indices, channels = self.select_channels_starting_by(channels_starting_by)
            micro_wire_to_keep.extend(indices)
            indices, channels = self.select_channels_with_exact_same_name_without_number(channels_without_number)
            micro_wire_to_keep.extend(indices)
            micro_wire_to_keep.extend(self.select_channels_with_exact_same_name_with_number(channels_with_number))

            # remove redondant microwire and sort them
            micro_wire_to_keep = np.unique(micro_wire_to_keep)
            # then we check if all the micro_wire data are available
            to_del = np.setdiff1d(micro_wire_to_keep, self.available_micro_wires)
            if len(to_del) > 0:
                for d in to_del:
                    micro_wire_to_keep = micro_wire_to_keep[micro_wire_to_keep != d]

        channels_to_keep = [self.channel_info_by_microwire[micro_wire] for micro_wire in micro_wire_to_keep]

        sleep_stages_to_keep = []
        if sleep_stage_indices is not None:
            for index in sleep_stage_indices:
                sleep_stages_to_keep.append(self.sleep_stages[index])

        if sleep_stage_selection is not None:
            sleep_stages_to_keep.extend(self.selection_sleep_stage_by_stage(sleep_stage_selection))

        if len(sleep_stages_to_keep) == 0:
            # In case no stage have been selected, then we put all stages in the order they were recorded
            sleep_stages_to_keep = self.sleep_stages

        # selecting spikes that happen during the time interval of selected sleep stages
        # in order to plot a raster plot, a start time and end time is needed
        # so for each stage selected, we should keep the timestamp of the first spike and the timestamp of the
        # last spike

        # first we count how many spike_trains (how many SU & MU),
        nb_units_spike_nums = 0
        for mw_index, micro_wire in enumerate(micro_wire_to_keep):
            if only_SU_and_MU:
                nb_units_to_keep = 0
                cluster_infos = self.cluster_info[micro_wire][0]
                for unit_cluster, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                    cluster = cluster_infos[unit_cluster]
                    if (cluster < 1) or (cluster > 2):
                        continue
                    # 1 == MU, 2 == SU
                    if keeping_only_SU:
                        if cluster == 1:
                            # not taking into consideraiton MU
                            continue
                    # looking if there are spiking
                    at_least_a_spike = False
                    if selection_range_time is not None:
                        start_time = selection_range_time[0]
                        stop_time = selection_range_time[1]
                        spikes_time = np.copy(spikes_time)
                        spikes_time = spikes_time[spikes_time >= start_time]
                        spikes_time = spikes_time[spikes_time <= stop_time]
                        if len(spikes_time) > 0:
                            at_least_a_spike = True
                    else:
                        for ss in sleep_stages_to_keep:
                            start_time = ss.start_time
                            stop_time = ss.stop_time
                            spikes_time = np.copy(spikes_time)
                            spikes_time = spikes_time[spikes_time >= start_time]
                            spikes_time = spikes_time[spikes_time <= stop_time]
                            if len(spikes_time) > 0:
                                at_least_a_spike = True
                                break
                    # counting it only if there is some spike during that interval
                    if at_least_a_spike:
                        nb_units_to_keep += 1

                nb_units_spike_nums += nb_units_to_keep
            else:
                nb_units_spike_nums += len(self.spikes_time_by_microwire[micro_wire])

        spike_trains = [np.zeros(0)] * nb_units_spike_nums

        # used to labels the ticks
        micro_wire_labels = []
        cluster_labels = []
        # cluster_indices indicate what is the index of the units among the clusters (indexing starts at 0)
        cluster_indices = []

        if selection_range_time is not None:
            start_time = selection_range_time[0]
            stop_time = selection_range_time[1]
            time_epochs = [(start_time, stop_time)]
        else:
            time_epochs = [(ss.start_time, ss.stop_time) for ss in sleep_stages_to_keep]

        for time_epoch in time_epochs:
            start_time = time_epoch[0]
            stop_time = time_epoch[1]

            unit_index = 0
            for mw_index, micro_wire in enumerate(micro_wire_to_keep):
                cluster_infos = self.cluster_info[micro_wire][0]
                for unit_cluster_index, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                    cluster = cluster_infos[unit_cluster_index]
                    # not taking into consideration artifact or non clustered
                    if (cluster < 1) or (cluster > 2):
                        continue
                    if keeping_only_SU:
                        if cluster == 1:
                            # not taking into consideraiton MU
                            continue
                    spikes_time = np.copy(spikes_time)
                    spikes_time = spikes_time[spikes_time >= start_time]
                    spikes_time = spikes_time[spikes_time <= stop_time]

                    if len(spikes_time) == 0:
                        # if no spikes we don't keep it
                        continue

                    if len(spike_trains[unit_index]) == 0:
                        spike_trains[unit_index] = spikes_time
                    else:
                        spike_trains[unit_index] = np.concatenate((spike_trains[unit_index], spikes_time))

                    micro_wire_labels.append(micro_wire)
                    cluster_labels.append(cluster)
                    cluster_indices.append(unit_cluster_index)

                    unit_index += 1

        # 1 = MU  2 = SU -1 = Artif.
        # 0 = Unassigned (is ignored)

        spike_struct = SpikeStructure(patient=self, spike_trains=spike_trains,
                                      microwire_labels=micro_wire_labels,
                                      cluster_labels=cluster_labels,
                                      title=title, cluster_indices=cluster_indices)
        # print(f"End of construct_spike_structure for {self.patient_id}")
        return spike_struct

    def select_channels_starting_by(self, channels_starting_by):
        """

        :param channels_starting_by: list of str, if empty list, return empty list, otherwise take the one starting
        with the same
        name (like "RAH" take RAH1, RAH2 etc...., if just "R" take all microwire on the right)
        :return:
        """
        result_indices = []
        result_channels = []
        for channel in channels_starting_by:
            result_indices.extend([i for i, ch in enumerate(self.channel_info_by_microwire)
                                   if isinstance(ch, str) and ch.startswith(channel)])
            result_channels.extend([ch for ch in self.channel_info_by_microwire
                                    if isinstance(ch, str) and ch.startswith(channel)])
        return result_indices, result_channels

    def select_channels_with_exact_same_name_without_number(self, channels):
        """
        Select channels without the precise index, for example select all "A" (all amygdala channels)
        :param channels: list of str:  full name without numbers
        :return:
        """
        result_indices = []
        result_channels = []
        for channel in channels:
            result_indices.extend([i for i, ch in enumerate(self.channel_info_by_microwire)
                                   if (ch.startswith(channel) and (len(ch) == (len(channel) + 1)))])
            result_channels.extend([ch for ch in self.channel_info_by_microwire
                                    if (ch.startswith(channel) and (len(ch) == (len(channel) + 1)))])
        return result_indices, result_channels

    def select_channels_with_exact_same_name_with_number(self, channels):
        """
        Select channels with the precise index, for example select all "A1" (amygdala channel 1)
        :param channels: list of full name with numbers
        :return:
        """
        result = []
        result.extend([i for i, ch in enumerate(self.channel_info_by_microwire)
                       if ch in channels])
        return result

    def selection_sleep_stage_by_stage(self, sleep_stage_selection):
        """

        :param sleep_stage_selection: list of str
        :return:
        """

        return [ss for ss in self.sleep_stages if ss.sleep_stage in sleep_stage_selection]

    def get_indices_of_sleep_stage(self, sleep_stage_name):
        return [i for i, ss in enumerate(self.sleep_stages) if ss.sleep_stage == sleep_stage_name]

    def descriptive_stats(self):
        """
        Print some descriptive stats about a patient
        :return:
        """
        for channels_starting_by in [None, "L", "R"]:
            n_su = 0
            n_mu = 0
            micro_wire_to_keep = []
            if channels_starting_by is None:
                micro_wire_to_keep = self.available_micro_wires
                print(f"n units: {len(micro_wire_to_keep)}")
                print(f"n invariant units: {len(self.is_invariant_units_dict)}")
                print(f"n responsive units: {len(self.is_responsive_units_dict)}")
            else:
                indices, channels = self.select_channels_starting_by(channels_starting_by)

                micro_wire_to_keep.extend(indices)
                # remove redondant microwire and sort them
                micro_wire_to_keep = np.unique(micro_wire_to_keep)
                # then we check if all the micro_wire data are available
                to_del = np.setdiff1d(micro_wire_to_keep, self.available_micro_wires)
                if len(to_del) > 0:
                    for d in to_del:
                        micro_wire_to_keep = micro_wire_to_keep[micro_wire_to_keep != d]

                # print(f"n units in {channels_starting_by}: {len(micro_wire_to_keep)}")

                invariant_keys = list(self.is_invariant_units_dict.keys())
                responsive_keys = list(self.is_responsive_units_dict.keys())

                print(f"n invariant units: {len([k for k in invariant_keys if channels_starting_by in k])}")
                print(f"n responsive units: {len([k for k in responsive_keys if channels_starting_by in k])}")
            mu_by_area_count = SortedDict()
            su_by_area_count = SortedDict()
            # A	AH	EC	MH	PH	PHC
            # print(f"self.channel_info_by_microwire {self.channel_info_by_microwire}")
            # print(f"self.available_micro_wires {self.available_micro_wires}")
            for micro_wire in micro_wire_to_keep:
                cluster_infos = self.cluster_info[micro_wire][0]
                for unit_cluster, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                    cluster = cluster_infos[unit_cluster]
                    if (cluster < 1) or (cluster > 2):
                        continue
                    if cluster == 1:
                        # == MU
                        n_mu += 1
                        counter_dict = mu_by_area_count
                    else:
                        n_su += 1
                        counter_dict = su_by_area_count
                    channel_name = self.channel_info_by_microwire[micro_wire]
                    # print(f'channel_name {channel_name}')
                    unique_channels = ["EC", "AH", "MH", "PHC"]
                    for channel in unique_channels:
                        if channel in channel_name:
                            counter_dict[channel] = counter_dict.get(channel, 0) + 1
                    if ("A" in channel_name) and ("AH" not in channel_name):
                        counter_dict["A"] = counter_dict.get("A", 0) + 1
                    if ("PH" in channel_name) and ("PHC" not in channel_name):
                        counter_dict["PH"] = counter_dict.get("PH", 0) + 1
            if channels_starting_by is None:
                print(f"From both side: n_su {n_su}, n_mu {n_mu}")
            else:
                print(f"For side {channels_starting_by}: n_su {n_su}, n_mu {n_mu}, total {n_su+n_mu}")
            print(f"mu_by_area_count: {mu_by_area_count}")
            print(f"su_by_area_count: {su_by_area_count}")
            print("")

        if len(self.stimuli_name_dict) > 0:
            print(f"Stimuli content: {self.stimuli_name_dict}")
            print(" ")

        print("sleep stages: ")
        for sleep_stage in self.sleep_stages:
            print(sleep_stage)

    @property
    def identifier(self):
        return self._identifier
