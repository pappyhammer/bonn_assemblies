import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import hdf5storage
from datetime import datetime
import os
# to add homemade package, go to preferences, then project interpreter, then click on the wheel symbol
# then show all, then select the interpreter and lick on the more right icon to display a list of folder and
# add the one containing the folder pattern_discovery
from pattern_discovery.seq_solver.markov_way import MarkovParameters
from pattern_discovery.seq_solver.markov_way import find_significant_patterns
import pattern_discovery.tools.param as p_disc_param
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.display.raster import plot_sum_active_clusters
from pattern_discovery.display.raster import plot_dendogram_from_fca
from pattern_discovery.display.misc import plot_hist_clusters_by_sce
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.trains as trains_module
from pattern_discovery.seq_solver.markov_way import order_spike_nums_by_seq
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.fca.fca import compute_and_plot_clusters_raster_fca_version
import pattern_discovery.clustering.fca.fca as fca
from pattern_discovery.clustering.kmean_version.k_mean_clustering import compute_and_plot_clusters_raster_kmean_version
from pattern_discovery.seq_solver.markov_way import find_sequences_in_ordered_spike_nums
from pattern_discovery.display.misc import plot_box_plots

import neo
import quantities as pq
import elephant.conversion as elephant_conv
from elephant.spike_train_correlation import corrcoef
# import elephant.cell_assembly_detection as cad
from cell_assembly_detection import cell_assembly_detection


# TODO: see to use scipy.sparse in the future

class BonnParameters(MarkovParameters):
    def __init__(self, path_results, time_str, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 max_branches, stop_if_twin, no_reverse_seq, error_rate, spike_rate_weight,
                 bin_size=1):
        super().__init__(time_inter_seq=time_inter_seq, min_duration_intra_seq=min_duration_intra_seq,
                         min_len_seq=min_len_seq, min_rep_nb=min_rep_nb, no_reverse_seq=no_reverse_seq,
                         max_branches=max_branches, stop_if_twin=stop_if_twin, error_rate=error_rate,
                         spike_rate_weight=spike_rate_weight,
                         bin_size=bin_size, path_results=path_results, time_str=time_str)
        self.sampling_rate = 32768


# frequency sampling is ~32 kHz (32.768 Hz)

class SleepStage:
    def __init__(self, number, start_time, stop_time, sleep_stage, conversion_datetime, conversion_timestamp):
        # timstamps are float, it's needed to multiple by 10^3 to get the real value, with represents microseconds
        self.start_time = start_time * 1000
        self.stop_time = stop_time * 1000
        # duration is in microseconds
        self.duration = self.stop_time - self.start_time
        self.sleep_stage = sleep_stage
        self.conversion_datetime = conversion_datetime
        self.conversion_timestamp = conversion_timestamp * 1000
        self.number = number

    def __str__(self):
        result = ""
        result += f"num  {self.number}, "
        result += f"sleep_stage  {self.sleep_stage}, "
        result += f"start_time  {self.start_time}, "
        result += f"stop_time  {self.stop_time}, \n"
        result += f"duration (usec)  {self.duration}, "
        result += f"duration (sec)  {self.duration / 1000000}, "
        result += f"duration (min)  {(self.duration / 1000000) / 60}"
        result += f",\n conversion_datetime  {self.conversion_datetime}, "
        result += f"conversion_timestamp  {self.conversion_timestamp}, "
        return result


class SpikeStructure:

    def __init__(self, patient, spike_nums, spike_trains, microwire_labels, cluster_labels, activity_threshold=None,
                 title=None, ordered_indices=None, ordered_spike_data=None,
                 one_sec=10 ** 6):
        self.patient = patient
        self.spike_nums = spike_nums
        self.spike_trains = spike_trains
        self.ordered_spike_data = ordered_spike_data
        # array of int
        self.microwire_labels = np.array(microwire_labels)
        # array of int
        self.cluster_labels = np.array(cluster_labels)
        self.activity_threshold = activity_threshold
        self.title = title
        self.labels = self.get_labels()
        self.ordered_indices = ordered_indices
        self.ordered_labels = None
        # one_sec reprents the number of times in one sec
        self.one_sec = one_sec
        if self.ordered_indices is not None:
            if self.spike_nums is not None:
                self.ordered_spike_nums = np.copy(self.spike_nums[ordered_indices, :])
            else:
                self.ordered_spike_nums = None
            if self.spike_trains is not None:
                self.ordered_spike_trains = []
                for index in ordered_indices:
                    self.ordered_spike_trains.append(self.spike_trains[index])
            else:
                self.ordered_spike_trains = None
            self.ordered_labels = []
            # y_ticks_labels_ordered = spike_nums_struct.labels[best_seq]
            for old_cell_index in self.ordered_indices:
                self.ordered_labels.append(self.labels[old_cell_index])

    def decrease_resolution(self, n=0, max_time=0):
        if n > 0:
            self.one_sec = self.one_sec * 10 ** -n
            if self.spike_nums is not None:
                n_times = self.spike_nums.shape[1]
                new_spike_nums = np.zeros((self.spike_nums.shape[0], int(np.ceil(n_times * 10 ** -n))), dtype="uint8")
                # print(f"new_spike_nums.shape {new_spike_nums.shape}")
                for cell, spike_train in enumerate(self.spike_nums):
                    # getting indices of where the cell is spiking
                    spike_train = np.where(spike_train)[0]
                    # then reducing dimension
                    spike_train = (spike_train * 10 ** -n).astype(int)
                    new_spike_nums[cell, spike_train] = 1
                self.spike_nums = new_spike_nums
            if self.spike_trains is not None:
                for i, spike_train in enumerate(self.spike_trains):
                    self.spike_trains[i] = spike_train * 10 ** -n
            return

        if max_time > 0:
            if self.spike_nums is not None:
                if max_time > 0:
                    max_time = 10 ** max_time
                    # first looking for the max value
                    max_value = np.max(self.spike_nums)

                    if max_value > max_time:
                        n_p = 0
                        while max_value > max_time:
                            max_value = max_value / 10
                            n_p += 1
                        self.decrease_resolution(n=n_p)
            elif self.spike_trains is not None:
                max_time = 10 ** max_time
                # first looking for the max value
                max_value = 0
                for i, spike_train in enumerate(self.spike_trains):
                    max_value = np.max((max_value, np.max(spike_train)))
                if max_value > max_time:
                    n_p = 0
                    while max_value > max_time:
                        max_value = max_value / 10
                        n_p += 1
                    # print(f"n_p {n_p}")
                    # decreasing by 10**n_p factor
                    self.decrease_resolution(n=n_p)

        # if self.spike_trains is not None:
        #     self.decrease_resolution_spike_trains(n=n, max_time=max_time)
        # if self.spike_nums is not None:
        #     self.decrease_resolution_spike_nums(n=n, max_time=max_time)

    def decrease_resolution_spike_nums(self, n=0, max_time=0):
        """

        :param n: integer, will divide all values by 10**-n
        :param max_time:  will divide all values by 10**-N, with N the minimum integer such as the max timestamps value
               will be < 10**max_time
        :return:
                """
        # if n > 0:
        #     n_times = self.spike_nums.shape[1]
        #     new_spike_nums = np.zeros((self.spike_nums.shape[0], int(np.ceil(n_times * 10 ** -n))), dtype="uint8")
        #     # print(f"new_spike_nums.shape {new_spike_nums.shape}")
        #     for cell, spike_train in enumerate(self.spike_nums):
        #         # getting indices of where the cell is spiking
        #         spike_train = np.where(spike_train)[0]
        #         # then reducing dimension
        #         spike_train = (spike_train * 10 ** -n).astype(int)
        #         new_spike_nums[cell, spike_train] = 1
        #     self.spike_nums = new_spike_nums

        if max_time > 0:
            max_time = 10 ** max_time
            # first looking for the max value
            max_value = np.max(self.spike_nums)

            if max_value > max_time:
                n_p = 0
                while max_value > max_time:
                    max_value = max_value / 10
                    n_p += 1
                self.decrease_resolution_spike_nums(n=n_p)

    def decrease_resolution_spike_trains(self, n=0, max_time=0):
        """

               :param n: integer, will divide all values by 10**-n
               :param max_time:  will divide all values by 10**-N, with N the minimum integer such as the max timestamps value
               will be < 10**max_time
               :return:
        """
        if n > 0:
            for i, spike_train in enumerate(self.spike_trains):
                self.spike_trains[i] = spike_train * 10 ** -n
        if max_time > 0:
            max_time = 10 ** max_time
            # first looking for the max value
            max_value = 0
            for i, spike_train in enumerate(self.spike_trains):
                max_value = np.max((max_value, np.max(spike_train)))
            if max_value > max_time:
                n_p = 0
                while max_value > max_time:
                    max_value = max_value / 10
                    n_p += 1
                # print(f"n_p {n_p}")
                # decreasing by 10**n_p factor
                self.decrease_resolution_spike_trains(n=n_p)

    def get_labels(self):
        labels = []
        cluster_to_label = {1: "MU ", 2: "SU ", -1: "Artif ", 0: ""}
        for i, micro_wire in enumerate(self.microwire_labels):
            channel = self.patient.channel_info_by_microwire[micro_wire]
            labels.append(f"{cluster_to_label[self.cluster_labels[i]]}{micro_wire} "
                          f"{channel}")
        return labels

    def get_nb_times_by_ms(self, nb_ms, as_int=False):
        result = (nb_ms * self.one_sec) / 1000
        if as_int:
            return int(result)
        return result

    def set_order(self, ordered_indices):
        if ordered_indices is None:
            self.ordered_spike_nums = np.copy(self.spike_nums)
        else:
            if self.spike_nums is not None:
                self.ordered_spike_nums = np.copy(self.spike_nums[ordered_indices, :])
            # else:
            #     self.ordered_spike_nums = None
            if self.spike_trains is not None:
                self.ordered_spike_trains = []
                for index in ordered_indices:
                    self.ordered_spike_trains.append(self.spike_trains[index])
            # else:
            #     self.ordered_spike_trains = None
            self.ordered_indices = ordered_indices
            self.ordered_labels = []
            for old_cell_index in self.ordered_indices:
                self.ordered_labels.append(self.labels[old_cell_index])


class SpikeTrainsStructure(SpikeStructure):
    """
    Class use to store the data from spike_nums analyses
    """

    def __init__(self, patient, spike_trains, microwire_labels, cluster_labels, activity_threshold=None,
                 title=None, ordered_spike_trains=None, ordered_indices=None,
                 one_sec=10 ** 6):
        super().__init__(patient=patient, spike_trains=spike_trains, microwire_labels=microwire_labels,
                         cluster_labels=cluster_labels,
                         activity_threshold=activity_threshold, title=title,
                         ordered_indices=ordered_indices, ordered_spike_data=ordered_spike_trains,
                         one_sec=one_sec)

    def get_spike_nums_structure(self):
        # TODO: return a SpikeNumsStructure
        pass

    def save_data(self, saving_path=None):
        if saving_path is None:
            saving_path = self.patient.param.path_results
        np.savez(f'{saving_path}/{self.title}_spike_nums_ordered_{self.patient.patient_id}.npz',
                 spike_trains=self.spike_trains, microwire_labels=self.microwire_labels,
                 ordered_spike_trains=self.ordered_spike_data,
                 ordered_indices=self.ordered_indices, activity_threshold=self.activity_threshold,
                 cluster_labels=self.cluster_labels)


class SpikeNumsStructure(SpikeStructure):
    """
    Class use to store the data from spike_nums analyses
    """

    def __init__(self, patient, spike_nums, microwire_labels, cluster_labels, activity_threshold=None,
                 title=None, ordered_spike_nums=None, ordered_indices=None, one_sec=10 ** 6):

        super().__init__(patient=patient, spike_nums=spike_nums, microwire_labels=microwire_labels,
                         cluster_labels=cluster_labels,
                         activity_threshold=activity_threshold, title=title,
                         ordered_indices=ordered_indices, ordered_spike_data=ordered_spike_nums,
                         one_sec=one_sec)

    def save_data(self, saving_path=None):
        if saving_path is None:
            saving_path = self.patient.param.path_results
        np.savez(f'{saving_path}/{self.title}_spike_nums_ordered_{self.patient.patient_id}.npz',
                 spike_nums=self.spike_nums, microwire_labels=self.microwire_labels,
                 ordered_spike_nums=self.ordered_spike_data,
                 ordered_indices=self.ordered_indices, activity_threshold=self.activity_threshold,
                 cluster_labels=self.cluster_labels)

    def get_spike_train_structure(self):
        # TODO: return a SpikeTrainStructure
        pass

    @staticmethod
    def load_from_data(filename, patient, one_sec=10 ** 6):
        # TODO: to update
        npzfile = np.load(filename)

        spike_nums = npzfile['spike_nums']
        microwire_labels = npzfile['microwire_labels']
        cluster_labels = npzfile['cluster_labels']
        activity_threshold = npzfile['activity_threshold']
        if not isinstance(activity_threshold, int):
            activity_threshold = None
        ordered_spike_nums = npzfile['ordered_spike_nums']
        ordered_indices = npzfile['ordered_indices']

        return SpikeNumsStructure(spike_nums=spike_nums, microwire_labels=microwire_labels,
                                  cluster_labels=cluster_labels, activity_threshold=activity_threshold,
                                  ordered_spike_nums=ordered_spike_nums, ordered_indices=ordered_indices,
                                  patient=patient, one_sec=one_sec)


class BonnPatient:
    def __init__(self, data_path, patient_id, param):
        self.patient_id = patient_id
        self.data_path = data_path
        self.param = param
        # number of units, one Multi-unit count as one unit
        self.n_units = 0

        # Filter the items and only keep files (strip out directories)
        files_in_dir = [item for item in os.listdir(data_path + patient_id + "/")
                        if os.path.isfile(os.path.join(data_path + patient_id + "/", item))]

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

        cluster_info_file = hdf5storage.loadmat(data_path + patient_id + "/" + "cluster_info.mat")
        label_info = cluster_info_file["label_info"]
        # contains whether an empty list if no cluster, or a list containing a list containing the type of cluster
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

        sleep_stages_file = hdf5storage.loadmat(data_path + patient_id + "/" + patient_id + "_sleepstages.mat",
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
        # print(f"sleepstages[0]: {sleepstages[1]}")
        # print(f"conversion_datetime {conversion_datetime}")
        # print(f"conversion_timestamp {conversion_timestamp[0][0]}")
        # print(f"conversion_timestamp int ? {isinstance(conversion_timestamp[0][0], int)}")
        print(f"total duration (min): {(total_duration / 1000000) / 60}")
        # print_mat_file_content(sleep_stages_file)
        self.available_micro_wires = []
        for file_in_dir in files_in_dir:
            # return
            if file_in_dir.startswith("times_pos_CSC"):
                # -1 to start by 0, to respect other matrices order
                microwire_number = int(file_in_dir[13:-4]) - 1
                self.available_micro_wires.append(microwire_number)
                data_file = hdf5storage.loadmat(data_path + patient_id + "/" + file_in_dir)
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
                # not need anymore because we add 0
                # for i, cluster_ref in enumerate(original_spikes_cluster_by_microwire):
                #     if cluster_ref > 0:
                #     original_spikes_cluster_by_microwire[i] -= 1

                # print(f"{microwire_number}  spikes_cluster_by_microwire[microwire_number] after: "
                #       f"{spikes_cluster_by_microwire[microwire_number]}")
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
        self.n_microwires = len(self.spikes_by_microwire)
        self.available_micro_wires = np.array(self.available_micro_wires)

    def print_channel_list(self):
        for i, channel in enumerate(self.channel_info_by_microwire):
            print(f"micro {i} channel: {channel}")

    def elephant_cad(self, path_results, time_str, sliding_window_ms,
                     alpha_p_value,
                     do_filter_spike_trains, n_cells_min_in_ass_to_plot, with_concatenation=False,
                     all_sleep_stages_in_order=False, save_formats="pdf"):
        """

        Args:
            path_results:
            time_str:
            sliding_window_ms:
            do_filter_spike_trains: if True, keeps only cells that don't fire too much
            n_cells_min_in_ass_to_plot: min number of cell in a cell assembly to print the cell assembly
            keeping_only_SU:
            with_concatenation:
            all_sleep_stages_in_order:
            save_formats:

        Returns:

        """

        # TODO: add a mode to concatenate all SWS stages
        # must be >= 2
        maxlag = 2
        alpha = alpha_p_value  # 0.05
        test_fake_data = False

        new_dir = f"{self.patient_id}_maxlag_{maxlag}_bin_{sliding_window_ms}_{time_str}"
        path_results = os.path.join(path_results, new_dir)
        os.mkdir(path_results)
        with open(os.path.join(path_results, new_dir + "_log.txt"), "w", encoding='UTF-8') as file:
            if not with_concatenation:
                for keeping_only_SU in [True, False]:
                    units_str = "SU_and_MU"
                    if keeping_only_SU:
                        units_str = "only_SU"
                    # for each sleep stage, we print all units, the right hemisphere one and the left ones
                    for s_index in np.arange(self.nb_sleep_stages):
                        # TODO: concatenate all same stages
                        # TODO: concatenate several stages sleep next to each other
                        for side in ["Left", "Right", "Left-Right"]:
                            if side == "Left-Right":
                                spike_struct = self.construct_spike_structure(sleep_stage_indices=[s_index],
                                                                              title=f"index_{s_index}_ss_left_and_right_{units_str}",
                                                                              spike_trains_format=True,
                                                                              spike_nums_format=False,
                                                                              keeping_only_SU=keeping_only_SU,
                                                                              )
                            else:

                                spike_struct = self.construct_spike_structure(sleep_stage_indices=[s_index],
                                                                              channels_starting_by=[side[0]],
                                                                              title=f"index_{s_index}_ss_{side}_{units_str}",
                                                                              spike_trains_format=True,
                                                                              spike_nums_format=False,
                                                                              keeping_only_SU=keeping_only_SU
                                                                              )
                            sleep_stage = self.sleep_stages[s_index].sleep_stage

                            # first we create a spike_trains in the neo format
                            spike_trains = []
                            t_start = None
                            t_stop = None
                            for cell in np.arange(len(spike_struct.spike_trains)):
                                spike_train = spike_struct.spike_trains[cell]
                                # convert frames in ms
                                spike_train = spike_train / 1000
                                spike_trains.append(spike_train)
                                if t_start is None:
                                    t_start = spike_train[0]
                                else:
                                    t_start = min(t_start, spike_train[0])
                                if t_stop is None:
                                    t_stop = spike_train[-1]
                                else:
                                    t_stop = max(t_stop, spike_train[-1])

                            duration_sec = (t_stop - t_start) / 1000
                            cell_labels = spike_struct.labels
                            # filtering spike_trains
                            if do_filter_spike_trains:
                                filtered_spike_trains = []
                                filtered_cell_labels = []
                                for cell in np.arange(len(spike_trains)):
                                    spike_train = spike_trains[cell]
                                    n_spike_normalized = len(spike_train) / duration_sec
                                    # print(f"n spikes: {n_spike_normalized}")
                                    if n_spike_normalized <= 5:
                                        filtered_spike_trains.append(spike_train)
                                        filtered_cell_labels.append(cell_labels[cell])
                                spike_trains = filtered_spike_trains
                                cell_labels = filtered_cell_labels

                            n_cells = len(spike_trains)

                            if test_fake_data:
                                n_cells = 20
                                spike_trains = []
                                t_start = 0.
                                t_stop = 60 * 5 * 1000

                                for cell_index in np.arange(3):
                                    n_spikes = 50
                                    spike_train = np.zeros(n_spikes)

                                    for spike in range(n_spikes):
                                        spike_train[spike] = (spike + 1) * 5500 + np.random.randint(1, 80)

                                    spike_trains.append(spike_train)

                                for cell_index in np.arange(3, 20):
                                    n_spikes = 100
                                    spike_train = np.zeros(n_spikes)

                                    for spike in range(n_spikes):
                                        spike_train[spike] = (spike + 1) * 2900 + + np.random.randint(1, 80)

                                    spike_trains.append(spike_train)

                            neo_spike_trains = []
                            for cell in np.arange(n_cells):
                                spike_train = spike_trains[cell]
                                # print(f"n_spikes: {cell_labels[cell]}: {len(spike_train)}")
                                neo_spike_train = neo.SpikeTrain(times=spike_train, units='ms',
                                                                 t_start=t_start,
                                                                 t_stop=t_stop)
                                neo_spike_trains.append(neo_spike_train)

                            binsize = sliding_window_ms * pq.ms
                            # print(f'binsize {binsize}')
                            spike_trains_binned = elephant_conv.BinnedSpikeTrain(neo_spike_trains, binsize=binsize)
                            # crosscorrelograms of firing times
                            # (before spike sorting; lags -10 ms to 10 ms, bin size 0.5 ms)
                            spike_trains_binned_0_5 = elephant_conv.BinnedSpikeTrain(neo_spike_trains,
                                                                                     binsize=0.5 * pq.ms)
                            # Spike-count correlations
                            corr_coef_matrix = corrcoef(spike_trains_binned_0_5)
                            """
                               A list of lists for each spike train (i.e., rows of the binned matrix), 
                               that in turn contains for each spike the index into the binned matrix where this spike enters.
                            """
                            spike_indices_in_bins = spike_trains_binned.spike_indices
                            n_bins = len(spike_trains_binned.bin_edges)
                            patterns_cad = cell_assembly_detection(data=spike_trains_binned, maxlag=maxlag,
                                                                   same_config_cut=False,
                                                                   alpha=alpha,
                                                                   min_occ=1, significance_pruning=True,
                                                                   verbose=True)
                            # print(f"patterns_cad {patterns_cad}")
                            # patterns_cad[0]['times'] does correspond to the number of rep in 'signature' for the
                            # the max number of cells in this given assembly
                            # if len(patterns_cad) > 0:
                            #     print(f"len(patterns_cad[0]['times']) {len(patterns_cad[0]['times'])}")

                            title = f"Stage {sleep_stage} (index {s_index}) {units_str} {side} channels " \
                                    f"{self.patient_id}, duration {np.round(duration_sec, 3)}, n bins: {n_bins}"
                            # stat_stage = f"Duration  {np.round(self.sleep_stages[s_index].duration / 1000000, 3)} sec, " \
                            #     f"n bins: {n_bins}"
                            print("")
                            print(f"***********  {title}  ***********")
                            file.write('\n' + f"***********  {title}  ***********" + '\n')

                            # print(f"spike_indices_in_bins {spike_indices_in_bins}")
                            print(f"n assemblies : {len(patterns_cad)}")
                            file.write(f"n assemblies : {len(patterns_cad)}" + '\n')

                            biggest_cell_ass = 0
                            for ass_index, patterns in enumerate(patterns_cad):
                                print(f"########## assembly {ass_index} ##########")
                                file.write(f"########## assembly {ass_index} ##########" + '\n')

                                labels_assembly = []
                                for order_index, cell_index in enumerate(patterns['neurons']):
                                    labels_assembly.append(cell_labels[cell_index])
                                biggest_cell_ass = max(biggest_cell_ass, len(patterns['neurons']))

                                print(f"{len(patterns['neurons'])} neurons: {labels_assembly}")
                                file.write(f"{len(patterns['neurons'])} neurons: {labels_assembly}" + '\n')
                                print(f"lags {patterns['lags']}")
                                file.write(f"lags {patterns['lags']}" + '\n')

                                for order_index, cell_index in enumerate(patterns['neurons']):
                                    rep_ass = patterns['signature'][order_index][1]

                                    print(f"{cell_labels[cell_index]}: {rep_ass} rep in assembly "
                                          f"vs {len(spike_trains[cell_index])} spikes, "
                                          f"n bins {len(np.unique(spike_indices_in_bins[cell_index]))}")
                                    file.write(f"{cell_labels[cell_index]}: {rep_ass} rep in assembly "
                                               f"vs {len(spike_trains[cell_index])} spikes, "
                                               f"n bins {len(np.unique(spike_indices_in_bins[cell_index]))}" + '\n')
                                    if order_index < len(patterns['neurons']) - 1:
                                        next_cell_index = patterns['neurons'][order_index + 1]
                                        pearson_corr = corr_coef_matrix[cell_index, next_cell_index]
                                        print(f"Corr {cell_labels[cell_index]} vs {cell_labels[next_cell_index]}: "
                                              f"{np.round(pearson_corr, 4)}")
                                    # print(f", "
                                    #       f"n spikes: {len(spike_indices_in_bins[cell_index])}")

                            print(f"///// N cells max in a cell assembly: {biggest_cell_ass}")
                            file.write(f"///// N cells max in a cell assembly: {biggest_cell_ass}" + '\n')
                            print("")
                            file.write('\n')

                            """
                            patterns_cad
                            contains the assemblies detected for the binsize chosen each assembly is a dictionary with attributes: 
                            ‘neurons’ : vector of units taking part to the assembly
                    
                            (unit order correspond to the agglomeration order)
                    
                            ‘lag’ : vector of time lags lag[z] is the activation delay between
                            neurons[1] and neurons[z+1]
                    
                            ‘pvalue’ : vector of pvalues. pvalue[z] is the p-value of the
                            statistical test between performed adding neurons[z+1] to the neurons[1:z]
                    
                            ‘times’ : assembly activation time. It reports how many times the
                            complete assembly activates in that bin. time always refers to the activation of the first listed assembly element 
                            (neurons[1]), that doesn’t necessarily corresponds to the first unit firing. 
                            The format is identified by the variable bool_times_format.
                    
                            ‘signature’ : array of two entries (z,c). The first is the number of
                            neurons participating in the assembly (size), the second is number of assembly occurrences.
                            """

                            # -------------- figuring out the order in the cell assembly ------------- #
                            # for each repetition of the cell assembly, we keep the delay between cells
                            for pattern_index, patterns in enumerate(patterns_cad):
                                activation_orders = []
                                activation_diffs = []
                                activation_times = t_start + (patterns['times'] * int(binsize))
                                for activation_time in activation_times:
                                    activation_spike_times = []
                                    cells = np.array(patterns['neurons'])
                                    for cell_loop_index, cell in enumerate(cells):
                                        spike_train = np.array(spike_trains[cell])
                                        spike_index = np.searchsorted(spike_train, activation_time)
                                        spike_index = min(len(spike_train) - 1, spike_index)
                                        activation_spike_times.append(spike_train[spike_index])
                                    sorted_arg = np.argsort(activation_spike_times)
                                    labels_ordered = []
                                    for cell in cells[sorted_arg]:
                                        labels_ordered.append(cell_labels[cell])
                                    activation_orders.append(tuple(labels_ordered))
                                    # TODO: display the mean value of diff for each combinaison
                                    activation_diff = np.diff(np.array(activation_spike_times)[sorted_arg])
                                    # print(f"activation_diff {activation_diff}")
                                    # allows to know the number of unique element later
                                    activation_diffs.append(list(activation_diff))
                                # print(f"activation_diffs {activation_diffs}")
                                act_orders_dict = dict()
                                act_diffs_by_order_dict = dict()
                                for index, activation_order in enumerate(activation_orders):
                                    act_orders_dict[activation_order] = act_orders_dict.get(activation_order, 0) + 1
                                    if activation_order not in act_diffs_by_order_dict:
                                        # print(f"activation_diffs[index] {activation_diffs[index]}")
                                        diffs_list = [[one_diff] for one_diff in activation_diffs[index]]
                                        # print(f"diffs_list {diffs_list}")
                                        act_diffs_by_order_dict[activation_order] = diffs_list
                                    else:
                                        for index_diff, diffs_list in enumerate(
                                                act_diffs_by_order_dict[activation_order]):
                                            # print(f"diffs_list {diffs_list}")
                                            diffs_list.append(activation_diffs[index][index_diff])
                                # act_diffs_dict not useful
                                # act_diffs_dict = dict()
                                # for activation_diff in activation_diffs:
                                #     act_diffs_dict[activation_diff] = act_diffs_dict.get(activation_diff, 0) + 1
                                print(f"pattern_index {pattern_index}, len: {len(act_orders_dict)}")
                                file.write(f"pattern_index {pattern_index}, len: {len(act_orders_dict)}" + '\n')
                                # print(f"activation_orders count {act_orders_dict}")
                                # file.write(f"activation_orders count {act_orders_dict}" + '\n')
                                counts_arg_sorted = np.argsort(list(act_orders_dict.values()))
                                act_order_values = list(act_orders_dict.keys())
                                for act_order_index in counts_arg_sorted[::-1]:
                                    act_order = act_order_values[act_order_index]
                                    diff_lists = act_diffs_by_order_dict[act_order]
                                    to_print = f"{act_order}: rep {act_orders_dict[act_order]}, "
                                    for diff_list in diff_lists:
                                        mean_diffs = np.mean(diff_list)
                                        median_diffs = np.median(diff_list)
                                        std_diffs = np.std(diff_list)
                                        p25_diffs = np.percentile(diff_list, 25)
                                        p75_diffs = np.percentile(diff_list, 75)
                                        to_print = to_print + f"mean: {np.round(mean_diffs, 3)}, " + \
                                                   f"std: {np.round(std_diffs, 3)}. "
                                        to_print = to_print + f"median: {np.round(median_diffs, 3)}, " + \
                                                   f"p25: {np.round(p25_diffs, 3)}, " + \
                                                   f"p75: {np.round(p75_diffs, 3)}  ### "
                                    print(f"|| Stats diffs: {to_print}")
                                    print("")
                                    file.write(f"|| Stats diffs: {to_print}" + '\n')
                                # TODO: save them as numpy array if needed
                                # print(f"activation_diffs count {act_diffs_dict}")
                                # print(f"activation_diffs  {activation_diffs}")
                                # print("activation_orders,  activation_diffs")
                                # file.write("activation_orders,  activation_diffs" + '\n')
                                # act_str = ""
                                # for index in range(len(activation_orders)):
                                #     act_str = act_str + f"{activation_orders}, {activation_diffs} | "
                                # print(f"{act_str}")
                                # file.write(f"{act_str}" + '\n')
                                # print(f"")
                                # file.write('\n')

                            # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
                            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                                      '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
                            cells_to_highlight_colors = []
                            cells_to_highlight = []
                            cell_new_order = []
                            all_cells = np.arange(n_cells)
                            cell_index_so_far = 0
                            for ca_index, cell_assembly in enumerate(patterns_cad):
                                n_cells_in_ca = 0
                                for neu in cell_assembly['neurons']:
                                    # a cell can be in more than one assembly
                                    if neu not in cell_new_order:
                                        cell_new_order.append(neu)
                                        n_cells_in_ca += 1
                                cells_to_highlight.extend(
                                    np.arange(cell_index_so_far, cell_index_so_far + n_cells_in_ca))
                                cell_index_so_far += n_cells_in_ca
                                cells_to_highlight_colors.extend([colors[ca_index % len(colors)]] * n_cells_in_ca)

                            cell_new_order.extend(list(np.setdiff1d(all_cells, cell_new_order)))
                            cell_new_order = np.array(cell_new_order)
                            ordered_spike_trains = []
                            if test_fake_data:
                                y_labels = np.arange(n_cells)
                                cell_new_order = np.arange(n_cells)
                            else:
                                y_labels = []
                                for cell_index in cell_new_order:
                                    ordered_spike_trains.append(spike_trains[cell_index])
                                    y_labels.append(cell_labels[cell_index])

                            if test_fake_data:
                                background_color = "black"
                                fig, ax = plt.subplots(nrows=1, ncols=1,
                                                       figsize=(90, 30))
                                fig.patch.set_facecolor(background_color)
                                ax.set_facecolor(background_color)
                                for st_idx, spike_train in enumerate(spike_trains):
                                    new_st_idx = np.where(cell_new_order == st_idx)[0][0]
                                    ax.plot(spike_train, [new_st_idx] * len(spike_train), '|', color="white")

                                y_ticks_labels_color = "white"
                                x_ticks_labels_color = "white"
                                ax.yaxis.label.set_color(y_ticks_labels_color)
                                ax.xaxis.label.set_color(x_ticks_labels_color)
                                ax.tick_params(axis='y', colors=y_ticks_labels_color)
                                ax.tick_params(axis='x', colors=x_ticks_labels_color)
                                ax.set_yticks(np.arange(n_cells))
                                ax.set_yticklabels(y_labels[:])
                                plt.ylim([-1, n_cells])
                                plt.xlabel('time (ms)')
                                plt.ylabel('neurons ids')
                                plt.title(f"Fake")
                                show_fig = True
                                if show_fig:
                                    plt.show()

                            for pattern_index, patterns in enumerate(patterns_cad):
                                if len(patterns['neurons']) < n_cells_min_in_ass_to_plot:
                                    continue

                                cell_new_order = []
                                all_cells = np.arange(n_cells)
                                cell_new_order.extend(patterns['neurons'])
                                cell_new_order.extend(list(np.setdiff1d(all_cells, cell_new_order)))
                                cell_new_order = np.array(cell_new_order)

                                background_color = "black"
                                fig, ax = plt.subplots(nrows=1, ncols=1,
                                                       figsize=(90, 30))
                                fig.patch.set_facecolor(background_color)
                                ax.set_facecolor(background_color)
                                for neu in patterns['neurons']:
                                    new_neu_index = np.where(cell_new_order == neu)[0][0]
                                    # color = colors[pattern_index % len(colors)]
                                    color = "red"
                                    ax.plot(t_start + (patterns['times'] * int(binsize)),
                                            [new_neu_index] * len(patterns['times']), 'o', color=color)
                                    # Raster plot of the data
                                for st_idx, spike_train in enumerate(spike_trains):
                                    new_st_idx = np.where(cell_new_order == st_idx)[0][0]
                                    ax.plot(spike_train, [new_st_idx] * len(spike_train), '|', color="white")

                                y_ticks_labels_color = "white"
                                x_ticks_labels_color = "white"
                                ax.yaxis.label.set_color(y_ticks_labels_color)
                                ax.xaxis.label.set_color(x_ticks_labels_color)
                                ax.tick_params(axis='y', colors=y_ticks_labels_color)
                                ax.tick_params(axis='x', colors=x_ticks_labels_color)
                                ax.set_yticks(np.arange(n_cells))
                                ax.set_yticklabels(y_labels[:])
                                plt.ylim([-1, n_cells])
                                plt.xlabel('time (ms)')
                                plt.ylabel('neurons ids')
                                plt.title(f"Stage {sleep_stage} (index {s_index}) {units_str} {side} channels "
                                          f"{self.patient_id}_cluster_{pattern_index}")
                                show_fig = False
                                if show_fig:
                                    plt.show()
                                file_name = f"cad_elephant_in_{side}_raster_plot_" \
                                            f"stage_{sleep_stage}_index_{s_index}" \
                                            f"_{units_str}_{self.patient_id}" \
                                            f"bin{binsize}_maxlag_{maxlag}_pattern_{pattern_index}" \
                                            f"_{len(patterns['neurons'])}_cells"

                                if isinstance(save_formats, str):
                                    save_formats = [save_formats]

                                for save_format in save_formats:
                                    fig.savefig(f'{path_results}/{file_name}.{save_format}',
                                                format=f"{save_format}",
                                                facecolor=fig.get_facecolor())

                                plt.close()
                            if test_fake_data:
                                raise Exception("first try")

    def build_raster_for_each_stage_sleep(self, decrease_factor=4,
                                          with_ordering=True, sliding_window_ms=250,
                                          keeping_only_SU=False, with_concatenation=False,
                                          all_sleep_stages_in_order=False):
        """

        :param decrease_factor:
        :param with_ordering:
        :param sliding_window_ms:
        :param keeping_only_SU:
        :param with_concatenation:
        :param all_sleep_stages_in_order: we plot the one hour recording in the same plot
        :return:
        """
        if not with_concatenation:
            units_str = "SU_and_MU"
            if keeping_only_SU:
                units_str = "only_SU"
            # for each sleep stage, we print all units, the right hemisphere one and the left ones
            for s_index in np.arange(self.nb_sleep_stages):
                for side in ["Left", "Right", "Left-Right"]:
                    if side == "Left-Right":
                        spike_struct = self.construct_spike_structure(sleep_stage_indices=[s_index],
                                                                      title=f"index_{s_index}_ss_left_and_right_{units_str}",
                                                                      spike_trains_format=True,
                                                                      spike_nums_format=False,
                                                                      keeping_only_SU=keeping_only_SU,
                                                                      )
                    else:

                        spike_struct = self.construct_spike_structure(sleep_stage_indices=[s_index],
                                                                      channels_starting_by=[side[0]],
                                                                      title=f"index_{s_index}_ss_{side}_{units_str}",
                                                                      spike_trains_format=True,
                                                                      spike_nums_format=False,
                                                                      keeping_only_SU=keeping_only_SU
                                                                      )
                    sleep_stage = self.sleep_stages[s_index].sleep_stage

                    spike_struct.decrease_resolution(n=decrease_factor)

                    # putting sliding window to 250 ms
                    sliding_window_duration = spike_struct.get_nb_times_by_ms(sliding_window_ms,
                                                                              as_int=True)

                    activity_threshold = get_sce_detection_threshold(spike_nums=spike_struct.spike_trains,
                                                                     window_duration=sliding_window_duration,
                                                                     spike_train_mode=True,
                                                                     n_surrogate=20,
                                                                     perc_threshold=95,
                                                                     debug_mode=False)
                    spike_struct.activity_threshold = activity_threshold
                    self.param.activity_threshold = activity_threshold

                    print("plot_spikes_raster")

                    if True:
                        plot_spikes_raster(spike_nums=spike_struct.spike_trains, param=self.param,
                                           spike_train_format=True,
                                           title=f"Stage {sleep_stage} (index {s_index}) {units_str} {side} channels "
                                                 f"{self.patient_id}",
                                           file_name=f"{side}_raster_plot_stage_{sleep_stage}_index_{s_index}"
                                                     f"_{units_str}_{self.patient_id}",
                                           y_ticks_labels=cell_labels,
                                           y_ticks_labels_size=4,
                                           save_raster=True,
                                           show_raster=False,
                                           plot_with_amplitude=False,
                                           activity_threshold=spike_struct.activity_threshold,
                                           # 500 ms window
                                           sliding_window_duration=sliding_window_duration,
                                           show_sum_spikes_as_percentage=True,
                                           spike_shape="|",
                                           spike_shape_size=1,
                                           save_formats="pdf")

                    # if with_ordering:
                    #     list_seq_dict, best_seq = order_spike_nums_by_seq(spike_nums_struct.spike_nums, self.param,
                    #                                                       with_printing=True)
                    #     # best_seq == corresponding_cells_index
                    #     ordered_spike_nums = np.copy(spike_nums_struct.spike_nums[best_seq, :])
                    #     spike_nums_struct.set_order(ordered_spike_nums=ordered_spike_nums, ordered_indices=best_seq)
                    #     spike_nums_struct.save_data()
                    #
                    #     plot_spikes_raster(spike_nums=spike_nums_struct.ordered_spike_nums, param=self.param,
                    #                        title=f"Stage {sleep_stage} (index {s_index}) {side} channels {self.patient_id} ordered",
                    #                        file_name=f"{side}_ordered_raster_plot_stage_{sleep_stage}_index_{s_index}_{self.patient_id}",
                    #                        y_ticks_labels=spike_nums_struct.ordered_labels,
                    #                        y_ticks_labels_size=4,
                    #                        save_raster=True,
                    #                        show_raster=False,
                    #                        sliding_window_duration=512,
                    #                        show_sum_spikes_as_percentage=True,
                    #                        plot_with_amplitude=False,
                    #                        activity_threshold=None,
                    #                        save_formats="pdf")
            return

        # concatenating all commomn sleep stage
        if all_sleep_stages_in_order:
            for side in ["Left-Right", "Left", "Right"]:
                if side == "Left-Right":
                    spike_nums_struct = self.construct_spike_structure(channels_starting_by=["R"],
                                                                       spike_nums_format=False)
                else:
                    spike_nums_struct = self.construct_spike_structure(channels_starting_by=[side[0]],
                                                                       spike_nums_format=False)

            plot_spikes_raster(spike_nums=spike_nums_struct.spike_trains, param=self.param,
                               spike_train_format=True,
                               title=f"All stages {side} channel {self.patient_id}",
                               file_name=f"{side}_raster_plot_stage_all_stages_{self.patient_id}",
                               y_ticks_labels=spike_nums_struct.labels,
                               y_ticks_labels_size=4,
                               save_raster=True,
                               show_raster=False,
                               sliding_window_duration=sliding_window_ms,
                               show_sum_spikes_as_percentage=True,
                               without_activity_sum=True,
                               plot_with_amplitude=False,
                               activity_threshold=None,
                               spike_shape="|",
                               spike_shape_size=1,
                               save_formats="pdf")
        else:
            for sleep_stage in ["1", "2", "3", "R", "W"]:
                for side in ["Left", "Right", "Left-Right"]:
                    if side == "Left-Right":
                        spike_nums_struct = self.construct_spike_structure(sleep_stage_selection=[sleep_stage],
                                                                           channels_starting_by=["R"],
                                                                           spike_nums_format=False)
                    else:
                        spike_nums_struct = self.construct_spike_structure(sleep_stage_selection=[sleep_stage],
                                                                           channels_starting_by=[side[0]],
                                                                           spike_nums_format=False)

                plot_spikes_raster(spike_nums=spike_nums_struct.spike_trains, param=self.param,
                                   spike_train_format=True,
                                   without_activity_sum=True,
                                   title=f"Stage {sleep_stage} {side} channel {self.patient_id}",
                                   file_name=f"{side}_raster_plot_stage_{sleep_stage}_{self.patient_id}",
                                   y_ticks_labels=spike_nums_struct.labels,
                                   y_ticks_labels_size=4,
                                   save_raster=True,
                                   show_raster=False,
                                   sliding_window_duration=512,
                                   show_sum_spikes_as_percentage=True,
                                   plot_with_amplitude=False,
                                   activity_threshold=None,
                                   save_formats="pdf")
            # if with_ordering:
            # list_seq_dict, best_seq = order_spike_nums_by_seq(spike_nums_struct.spike_nums, self.param,
            #                                                   with_printing=True)
            # # best_seq == corresponding_cells_index
            # ordered_spike_nums = np.copy(spike_nums_struct.spike_nums[best_seq, :])
            # spike_nums_struct.set_order(ordered_spike_nums=ordered_spike_nums, ordered_indices=best_seq)
            # spike_nums_struct.save_data()
            #
            # plot_spikes_raster(spike_nums=spike_nums_struct.ordered_spike_nums, param=self.param,
            #                    title=f"Stage {sleep_stage} {side} channel {self.patient_id} ordered",
            #                    file_name=f"{side}_ordered_raster_plot_stage_{sleep_stage}_{self.patient_id}",
            #                    y_ticks_labels=spike_nums_struct.ordered_labels,
            #                    y_ticks_labels_size=4,
            #                    save_raster=True,
            #                    show_raster=False,
            #                    sliding_window_duration=512,
            #                    show_sum_spikes_as_percentage=True,
            #                    plot_with_amplitude=False,
            #                    activity_threshold=None,
            #                    save_formats="pdf")

    def print_sleep_stages_info(self, selected_indices=None):
        if selected_indices is None:
            for ss in self.sleep_stages:
                print(f"{ss}\n")
        else:
            for index in selected_indices:
                print(f"{self.sleep_stages[index]}\n")

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
                                   if ch.startswith(channel)])
            result_channels.extend([ch for ch in self.channel_info_by_microwire
                                    if ch.startswith(channel)])
        return result_indices, result_channels

    def select_channels_with_exact_same_name_without_number(self, channels):
        """

        :param channels: list of full name without numbers
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

        :param channels_starting_by: list of full name with numbers
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

        print(f"descriptive_stats for {self.patient_id}")

        for channels_starting_by in [None, "L", "R"]:
            n_su = 0
            n_mu = 0
            micro_wire_to_keep = []
            if (channels_starting_by is None):
                micro_wire_to_keep = self.available_micro_wires
                print(f"n microwires: {len(micro_wire_to_keep)}")
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

                print(f"n microwiresin {channels_starting_by}: {len(micro_wire_to_keep)}")
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

            print(f"For side {channels_starting_by}: n_su {n_su}, n_mu {n_mu}")
            print(f"mu_by_area_count: {mu_by_area_count}")
            print(f"su_by_area_count: {su_by_area_count}")
            print("")

        print("sleep stages: ")
        for sleep_stage in self.sleep_stages:
            print(sleep_stage)

    def are_stages_concatenable(self, ss_1, ss_2):
        """
        Check if we can  concatenate 2 stages
        Args:
            ss_1:
            ss_2:
            join_sws:

        Returns:

        """
        # only stage 1, 2 and 3 could be next to each other
        if ss_1.sleep_stage in ['123']:
            if ss_2.sleep_stage in ['123']:
                return True
            else:
                return False
        else:
            return False

    def construct_spike_structure_new_version(self, spike_trains_format=True, spike_nums_format=True,
                                              max_duration_in_sec=240,
                                              join_sws=True,
                                              sleep_stage_indices=None,
                                              sleep_stage_selection=None, channels_starting_by=None,
                                              channels_without_number=None, channels_with_number=None,
                                              title=None, keeping_only_SU=False):
        """

        :param sleep_stage_index: list of sleep_stage_index
        :param join_sws: means all stages that are not R and W, will be considered the same
        :param spike_trains_format: if True, will construct a spike_trains into Spiketructure
        :param spike_nums_format: if True, will construct a spike_nums SpikeStructure
        :param channels: list of str, if empty list, take them all, otherwise take the one starting with the same
        name (like "RAH" take RAH1, RAH2 etc...., if just "R" take all microwire on the right)
        :param channels_to_study: full name without numbers
        :return: list of spike_struct
        """
        # print(f"construct_spike_structure start for {self.patient_id}")
        # don't put non-assigned clusters
        only_SU_and_MU = True
        # toto
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
        # print(f"micro_wire_to_keep {micro_wire_to_keep}")
        channels_to_keep = [self.channel_info_by_microwire[micro_wire] for micro_wire in micro_wire_to_keep]
        # if (sleep_stage_indices is None) and (sleep_stage_selection is None):
        #     return

        sleep_stages_to_keep = []
        if sleep_stage_indices is not None:
            for index in sleep_stage_indices:
                sleep_stages_to_keep.append(self.sleep_stages[index])

        if sleep_stage_selection is not None:
            sleep_stages_to_keep.extend(self.selection_sleep_stage_by_stage(sleep_stage_selection))

        if len(sleep_stages_to_keep) == 0:
            # then we put all stages in the order they were recorded
            sleep_stages_to_keep = self.sleep_stages
        # for ss in sleep_stages_to_keep:
        #     print(ss)

        last_timestamps = None
        spike_struct_list = list()
        index_sleep_stage = 0
        min_duration_sec = 10

        while index_sleep_stage < len(sleep_stages_to_keep):

            n_stages_used = 0
            stages_used = []
            current_duration_sec = 0
            if last_timestamps is None:
                start_time = None
            else:
                start_time = last_timestamps + 1
            stop_time = None
            for ss_index, ss in enumerate(sleep_stages_to_keep[index_sleep_stage:]):
                ss_index = index_sleep_stage + ss_index
                stages_used.append(ss)
                # first we define the start_time and stop_time of to cover max_duration
                if start_time is None:
                    start_time = ss.start_time
                    # duration is in microseconds
                    duration = ss.stop_time - ss.start_time
                    duration_sec = duration / 1000000
                    if duration_sec < max_duration_in_sec:
                        current_duration_sec += duration_sec
                        n_stages_used += 1
                        # we checked if we can continue depending of the next sleep stage
                        if ss_index == (len(sleep_stages_to_keep) - 1):
                            continue
                        if not join_sws:
                            break
                        if self.are_stages_concatenable(ss_1=ss, ss_2=sleep_stages_to_keep[ss_index + 1]):
                            continue
                        else:
                            break

                    else:
                        current_duration_sec = max_duration_in_sec
                        stop_time = ss.start_time + (max_duration_in_sec * 1000000)
                        break
                else:
                    # start_time is not None
                    duration = ss.stop_time - start_time
                    duration_sec = duration / 1000000
                    if duration_sec < max_duration_in_sec:
                        current_duration_sec += duration_sec
                        n_stages_used += 1
                        # we checked if we can continue depending of the next sleep stage
                        if ss_index == (len(sleep_stages_to_keep) - 1):
                            continue
                        if not join_sws:
                            break
                        if self.are_stages_concatenable(ss_1=ss, ss_2=sleep_stages_to_keep[ss_index + 1]):
                            continue
                        else:
                            break
                    else:
                        current_duration_sec = max_duration_in_sec
                        stop_time = start_time + (max_duration_in_sec * 1000000)
                        break

            index_sleep_stage += n_stages_used

            last_timestamps = stop_time

            if current_duration_sec < min_duration_sec:
                # setting a min duration
                print("Min duration reached, skipping to next one")
                continue

            # selecting spikes that happen during the time interval of selected sleep stages
            # in order to plot a raster plot, a start time and end time is needed
            # so for each stage selected, we should keep the timestamp of the first spike and the timestamp of the
            # last spike

            # first containing how many lines on spike_nums,

            nb_units_spike_nums = 0
            for mw_index, micro_wire in enumerate(micro_wire_to_keep):
                if only_SU_and_MU:
                    nb_units_to_keep = 0
                    cluster_infos = self.cluster_info[micro_wire][0]
                    for unit_cluster, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                        cluster = cluster_infos[unit_cluster]
                        if (cluster < 1) or (cluster > 2):
                            continue
                        if keeping_only_SU:
                            if cluster == 1:
                                # not taking into consideraiton MU
                                continue
                        # looking if there are spiking
                        at_least_a_spike = False
                        spikes_time = np.copy(spikes_time)
                        spikes_time = spikes_time[spikes_time >= start_time]
                        spikes_time = spikes_time[spikes_time <= stop_time]
                        if len(spikes_time) > 0:
                            # counting it only if there is some spike during that interval
                            nb_units_to_keep += 1

                    nb_units_spike_nums += nb_units_to_keep
                else:
                    nb_units_spike_nums += len(self.spikes_time_by_microwire[micro_wire])

            spike_trains = None
            spike_nums = None
            if spike_trains_format:
                spike_trains = [np.zeros(0)] * nb_units_spike_nums
            if spike_nums_format:
                spike_nums = np.zeros((nb_units_spike_nums, 0), dtype="int8")

            # dict with key the microwire number and the value is a dict with keys the cluster position and value
            # are the spikes for each cluster
            micro_wires_spikes_time_stamps = dict()
            # correspondance between micro_wire number and their units and index in the spike_nums matrix
            micro_wires_spike_nums_index = dict()
            units_index = 0
            for mw_index, micro_wire in enumerate(micro_wire_to_keep):
                micro_wires_spike_nums_index[micro_wire] = dict()  # mw_index
                micro_wires_spikes_time_stamps[micro_wire] = dict()
                cluster_infos = self.cluster_info[micro_wire][0]
                for unit_cluster, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                    cluster = cluster_infos[unit_cluster]
                    # not taking into consideration artifact or non clustered
                    if (cluster < 1) or (cluster > 2):
                        continue
                    if keeping_only_SU:
                        if cluster == 1:
                            # not taking into consideraiton MU
                            continue
                    # spikes_time = np.copy(self.spikes_time_by_microwire[micro_wire])
                    # print(f"micro_wire {micro_wire}, spikes_time {spikes_time}")
                    spikes_time = np.copy(spikes_time)
                    spikes_time = spikes_time[spikes_time >= start_time]
                    spikes_time = spikes_time[spikes_time <= stop_time]
                    # print(f"filtered: micro_wire {micro_wire}, spikes_time {spikes_time}")
                    if len(spikes_time) > 0:
                        if (min_time is None) or (spikes_time[0] < min_time):
                            min_time = spikes_time[0]
                        if (max_time is None) or (spikes_time[-1] > max_time):
                            max_time = spikes_time[-1]
                    else:
                        # if no spikes we don't keep it
                        continue
                    micro_wires_spikes_time_stamps[micro_wire][unit_cluster] = spikes_time
                    # units_index represents the line number on spike_nums
                    micro_wires_spike_nums_index[micro_wire][unit_cluster] = units_index
                    units_index += 1

            if spike_trains_format:
                # keeping the original time_stamps, even for different sleep stages, means there is going to be
                # large gap if concatenation is done
                for micro_wire, units_time_stamps_dict in micro_wires_spikes_time_stamps.items():
                    for units_cluster_index, time_stamps in units_time_stamps_dict.items():
                        cluster_infos = self.cluster_info[micro_wire][0]
                        cluster = cluster_infos[units_cluster_index]
                        # to change
                        if only_SU_and_MU:
                            if (cluster < 1) or (cluster > 2):
                                continue

                            if keeping_only_SU:
                                if cluster == 1:
                                    # not taking into consideraiton MU
                                    continue
                        # micro_wires_spike_nums_index[micro_wire][units_cluster_index]
                        # corresponds to the line number of spike_nums
                        unit_index = micro_wires_spike_nums_index[micro_wire][units_cluster_index]
                        if len(spike_trains[unit_index]) == 0:
                            spike_trains[unit_index] = time_stamps
                        else:
                            spike_trains[unit_index] = np.concatenate((spike_trains[unit_index], time_stamps))
                        # print(f"{unit_index}: len(time_stamps) {len(time_stamps)}")

            if spike_nums_format:
                len_for_ss = int(max_time - min_time)
                # print(f"len_for_ss {len_for_ss}")
                # new index, using the min time stamp as a reference
                new_micro_wires_spikes_time_stamps = dict()
                for micro_wire, units_time_stamps_dict in micro_wires_spikes_time_stamps.items():
                    new_micro_wires_spikes_time_stamps[micro_wire] = dict()
                    for units_cluster_index, time_stamps in units_time_stamps_dict.items():
                        new_micro_wires_spikes_time_stamps[micro_wire][units_cluster_index] = time_stamps - min_time
                    # print(f"micro_wire {micro_wire}, time_stamps-min_time {time_stamps-min_time}")
                micro_wires_spikes_time_stamps = new_micro_wires_spikes_time_stamps

                print(f"nb_units_spike_nums {nb_units_spike_nums}, spike_nums.shape[1] {spike_nums.shape[1]}, "
                      f"len_for_ss {len_for_ss}")
                new_spike_nums = np.zeros((nb_units_spike_nums, (spike_nums.shape[1] + len_for_ss + 1)),
                                          dtype="int8")
                new_spike_nums[:, :spike_nums.shape[1]] = spike_nums
                for micro_wire, units_time_stamps_dict in micro_wires_spikes_time_stamps.items():
                    for units_cluster_index, time_stamps in units_time_stamps_dict.items():
                        cluster_infos = self.cluster_info[micro_wire][0]
                        cluster = cluster_infos[units_cluster_index]
                        # to change
                        if only_SU_and_MU:
                            if (cluster < 1) or (cluster > 2):
                                continue
                            if keeping_only_SU:
                                if cluster == 1:
                                    # not taking into consideraiton MU
                                    continue
                        # micro_wires_spike_nums_index[micro_wire][units_cluster_index]
                        # corresponds to the line number of spike_nums
                        new_spike_nums[micro_wires_spike_nums_index[micro_wire][units_cluster_index],
                                       spike_nums.shape[1] + time_stamps.astype(int)] = 1
                spike_nums = new_spike_nums

            # print(f"spike_nums.shape {spike_nums.shape}")
            # print(f"np.sum(spike_nums, axis=1) {np.sum(spike_nums, axis=1)}")
            # to display raster plot, a binage will be necessary

            # used to labels the ticks
            micro_wire_labels = []
            cluster_labels = []
            for i, micro_wire in enumerate(micro_wire_to_keep):
                for unit_cluster_index in self.spikes_time_by_microwire[micro_wire].keys():
                    cluster_infos = self.cluster_info[micro_wire][0]
                    cluster = cluster_infos[unit_cluster_index]
                    if only_SU_and_MU:
                        if (cluster < 1) or (cluster > 2):
                            continue
                        if keeping_only_SU:
                            if cluster == 1:
                                # not taking into consideraiton MU
                                continue
                    micro_wire_labels.append(micro_wire)
                    cluster_labels.append(cluster)

            # 1 = MU  2 = SU -1 = Artif.
            # 0 = Unassigned (is ignored)
            # if spike_trains_format:
            #     # first transforming list as np.array
            #     # for i, train in enumerate(spike_trains):
            #     #     spike_trains[i] = np.array(spike_trains[i])
            #     spike_struct = SpikeTrainsStructure(patient=self, spike_trains=spike_trains,
            #                                         microwire_labels=micro_wire_labels,
            #                                         cluster_labels=cluster_labels,
            #                                         title=title)
            # else:
            spike_struct = SpikeStructure(patient=self, spike_trains=spike_trains, spike_nums=spike_nums,
                                          microwire_labels=micro_wire_labels,
                                          cluster_labels=cluster_labels,
                                          title=title)
            spike_struct_list.append(spike_struct)
        # print(f"End of construct_spike_structure for {self.patient_id}")
        return spike_struct_list  # spike_nums, micro_wire_to_keep, channels_to_keep, labels

    def construct_spike_structure(self, spike_trains_format=True, spike_nums_format=True,
                                  sleep_stage_indices=None,
                                  sleep_stage_selection=None, channels_starting_by=None,
                                  channels_without_number=None, channels_with_number=None,
                                  title=None, keeping_only_SU=False):
        """

        :param sleep_stage_index: list of sleep_stage_index
        :param spike_trains_format: if True, will construct a spike_trains into Spiketructure
        :param spike_nums_format: if True, will construct a spike_nums SpikeStructure
        :param channels: list of str, if empty list, take them all, otherwise take the one starting with the same
        name (like "RAH" take RAH1, RAH2 etc...., if just "R" take all microwire on the right)
        :param channels_to_study: full name without numbers
        :return:
        """
        # print(f"construct_spike_structure start for {self.patient_id}")
        # don't put non-assigned clusters
        only_SU_and_MU = True
        # toto
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
        # print(f"micro_wire_to_keep {micro_wire_to_keep}")
        channels_to_keep = [self.channel_info_by_microwire[micro_wire] for micro_wire in micro_wire_to_keep]
        # if (sleep_stage_indices is None) and (sleep_stage_selection is None):
        #     return

        sleep_stages_to_keep = []
        if sleep_stage_indices is not None:
            for index in sleep_stage_indices:
                sleep_stages_to_keep.append(self.sleep_stages[index])

        if sleep_stage_selection is not None:
            sleep_stages_to_keep.extend(self.selection_sleep_stage_by_stage(sleep_stage_selection))

        if len(sleep_stages_to_keep) == 0:
            # then we put all stages in the order they were recorded
            sleep_stages_to_keep = self.sleep_stages
        # for ss in sleep_stages_to_keep:
        #     print(ss)

        # selecting spikes that happen during the time interval of selected sleep stages
        # in order to plot a raster plot, a start time and end time is needed
        # so for each stage selected, we should keep the timestamp of the first spike and the timestamp of the
        # last spike

        # first containing how many lines on spike_nums,

        nb_units_spike_nums = 0
        for mw_index, micro_wire in enumerate(micro_wire_to_keep):
            if only_SU_and_MU:
                nb_units_to_keep = 0
                cluster_infos = self.cluster_info[micro_wire][0]
                for unit_cluster, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                    cluster = cluster_infos[unit_cluster]
                    if (cluster < 1) or (cluster > 2):
                        continue
                    if keeping_only_SU:
                        if cluster == 1:
                            # not taking into consideraiton MU
                            continue
                    # looking if there are spiking
                    at_least_a_spike = False
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

        spike_trains = None
        spike_nums = None
        if spike_trains_format:
            spike_trains = [np.zeros(0)] * nb_units_spike_nums
        if spike_nums_format:
            spike_nums = np.zeros((nb_units_spike_nums, 0), dtype="int8")

        for ss in sleep_stages_to_keep:
            start_time = ss.start_time
            stop_time = ss.stop_time
            min_time = None
            max_time = None

            # dict with key the microwire number and the value is a dict with keys the cluster position and value
            # are the spikes for each cluster
            micro_wires_spikes_time_stamps = dict()
            # correspondance between micro_wire number and their units and index in the spike_nums matrix
            micro_wires_spike_nums_index = dict()
            units_index = 0
            for mw_index, micro_wire in enumerate(micro_wire_to_keep):
                micro_wires_spike_nums_index[micro_wire] = dict()  # mw_index
                micro_wires_spikes_time_stamps[micro_wire] = dict()
                cluster_infos = self.cluster_info[micro_wire][0]
                for unit_cluster, spikes_time in self.spikes_time_by_microwire[micro_wire].items():
                    cluster = cluster_infos[unit_cluster]
                    # not taking into consideration artifact or non clustered
                    if (cluster < 1) or (cluster > 2):
                        continue
                    if keeping_only_SU:
                        if cluster == 1:
                            # not taking into consideraiton MU
                            continue
                    # spikes_time = np.copy(self.spikes_time_by_microwire[micro_wire])
                    # print(f"micro_wire {micro_wire}, spikes_time {spikes_time}")
                    spikes_time = np.copy(spikes_time)
                    spikes_time = spikes_time[spikes_time >= start_time]
                    spikes_time = spikes_time[spikes_time <= stop_time]
                    # print(f"filtered: micro_wire {micro_wire}, spikes_time {spikes_time}")
                    if len(spikes_time) > 0:
                        if (min_time is None) or (spikes_time[0] < min_time):
                            min_time = spikes_time[0]
                        if (max_time is None) or (spikes_time[-1] > max_time):
                            max_time = spikes_time[-1]
                    else:
                        # if no spikes we don't keep it
                        continue
                    micro_wires_spikes_time_stamps[micro_wire][unit_cluster] = spikes_time
                    # units_index represents the line number on spike_nums
                    micro_wires_spike_nums_index[micro_wire][unit_cluster] = units_index
                    units_index += 1

            if spike_trains_format:
                # keeping the original time_stamps, even for different sleep stages, means there is going to be
                # large gap if concatanation is done
                for micro_wire, units_time_stamps_dict in micro_wires_spikes_time_stamps.items():
                    for units_cluster_index, time_stamps in units_time_stamps_dict.items():
                        cluster_infos = self.cluster_info[micro_wire][0]
                        cluster = cluster_infos[units_cluster_index]
                        # to change
                        if only_SU_and_MU:
                            if (cluster < 1) or (cluster > 2):
                                continue

                            if keeping_only_SU:
                                if cluster == 1:
                                    # not taking into consideraiton MU
                                    continue
                        # micro_wires_spike_nums_index[micro_wire][units_cluster_index]
                        # corresponds to the line number of spike_nums
                        unit_index = micro_wires_spike_nums_index[micro_wire][units_cluster_index]
                        if len(spike_trains[unit_index]) == 0:
                            spike_trains[unit_index] = time_stamps
                        else:
                            spike_trains[unit_index] = np.concatenate((spike_trains[unit_index], time_stamps))
                        # print(f"{unit_index}: len(time_stamps) {len(time_stamps)}")

            if spike_nums_format:
                len_for_ss = int(max_time - min_time)
                # print(f"len_for_ss {len_for_ss}")
                # new index, using the min time stamp as a reference
                new_micro_wires_spikes_time_stamps = dict()
                for micro_wire, units_time_stamps_dict in micro_wires_spikes_time_stamps.items():
                    new_micro_wires_spikes_time_stamps[micro_wire] = dict()
                    for units_cluster_index, time_stamps in units_time_stamps_dict.items():
                        new_micro_wires_spikes_time_stamps[micro_wire][units_cluster_index] = time_stamps - min_time
                    # print(f"micro_wire {micro_wire}, time_stamps-min_time {time_stamps-min_time}")
                micro_wires_spikes_time_stamps = new_micro_wires_spikes_time_stamps

                print(f"nb_units_spike_nums {nb_units_spike_nums}, spike_nums.shape[1] {spike_nums.shape[1]}, "
                      f"len_for_ss {len_for_ss}")
                new_spike_nums = np.zeros((nb_units_spike_nums, (spike_nums.shape[1] + len_for_ss + 1)), dtype="int8")
                new_spike_nums[:, :spike_nums.shape[1]] = spike_nums
                for micro_wire, units_time_stamps_dict in micro_wires_spikes_time_stamps.items():
                    for units_cluster_index, time_stamps in units_time_stamps_dict.items():
                        cluster_infos = self.cluster_info[micro_wire][0]
                        cluster = cluster_infos[units_cluster_index]
                        # to change
                        if only_SU_and_MU:
                            if (cluster < 1) or (cluster > 2):
                                continue
                            if keeping_only_SU:
                                if cluster == 1:
                                    # not taking into consideraiton MU
                                    continue
                        # micro_wires_spike_nums_index[micro_wire][units_cluster_index]
                        # corresponds to the line number of spike_nums
                        new_spike_nums[micro_wires_spike_nums_index[micro_wire][units_cluster_index],
                                       spike_nums.shape[1] + time_stamps.astype(int)] = 1
                spike_nums = new_spike_nums

        # print(f"spike_nums.shape {spike_nums.shape}")
        # print(f"np.sum(spike_nums, axis=1) {np.sum(spike_nums, axis=1)}")
        # to display raster plot, a binage will be necessary

        # used to labels the ticks
        micro_wire_labels = []
        cluster_labels = []
        for i, micro_wire in enumerate(micro_wire_to_keep):
            for unit_cluster_index in self.spikes_time_by_microwire[micro_wire].keys():
                cluster_infos = self.cluster_info[micro_wire][0]
                cluster = cluster_infos[unit_cluster_index]
                if only_SU_and_MU:
                    if (cluster < 1) or (cluster > 2):
                        continue
                    if keeping_only_SU:
                        if cluster == 1:
                            # not taking into consideraiton MU
                            continue
                micro_wire_labels.append(micro_wire)
                cluster_labels.append(cluster)

        # 1 = MU  2 = SU -1 = Artif.
        # 0 = Unassigned (is ignored)
        # if spike_trains_format:
        #     # first transforming list as np.array
        #     # for i, train in enumerate(spike_trains):
        #     #     spike_trains[i] = np.array(spike_trains[i])
        #     spike_struct = SpikeTrainsStructure(patient=self, spike_trains=spike_trains,
        #                                         microwire_labels=micro_wire_labels,
        #                                         cluster_labels=cluster_labels,
        #                                         title=title)
        # else:
        spike_struct = SpikeStructure(patient=self, spike_trains=spike_trains, spike_nums=spike_nums,
                                      microwire_labels=micro_wire_labels,
                                      cluster_labels=cluster_labels,
                                      title=title)
        # print(f"End of construct_spike_structure for {self.patient_id}")
        return spike_struct  # spike_nums, micro_wire_to_keep, channels_to_keep, labels


def print_mat_file_content(mat_file):
    print(f"{str(type(mat_file))}")  # ==> Out[9]: dict
    print(f"{'*' * 79}")
    for key, value in mat_file.items():
        print(f'{key} {type(value)}')
        print(f'np.shape(value): {np.shape(value)}')
        print(f"{'*' * 79}")


def show_plot_raster(spike_nums, patient, title=None, file_name=None,
                     save_raster=True,
                     show_raster=True):
    param = patient.param

    n_times = len(spike_nums[0, :])

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                   gridspec_kw={'height_ratios': [10, 2]},
                                   figsize=(15, 8))
    fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})

    ax1.set_facecolor('black')

    for y, neuron in enumerate(spike_nums):
        # print(f"Neuron {y}, total spikes {len(np.where(neuron)[0])}, "
        #       f"nb > 2: {len(np.where(neuron>2)[0])}, nb < 2: {len(np.where(neuron[neuron<2])[0])}")
        color_neuron = "white"

        ax1.vlines(np.where(neuron)[0], y - .5, y + .5, color=color_neuron, linewidth=0.3)

    ax1.set_ylim(-1, len(spike_nums))
    ax1.set_xlim(-1, len(spike_nums[0, :]) + 1)
    # ax1.margins(x=0, tight=True)

    if title is None:
        ax1.set_title('Spike raster plot')
    else:
        ax1.set_title(title)
    # Give x axis label for the spike raster plot
    # ax.xlabel('Frames')
    # Give y axis label for the spike raster plot
    ax1.set_ylabel('Cells (#)')

    binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
    for neuron, spikes in enumerate(spike_nums):
        binary_spikes[neuron, spikes > 0] = 1
    # if param.bin_size > 1:
    #     sum_spikes = np.mean(np.split(np.sum(binary_spikes, axis=0), n_times // param.bin_size), axis=1)
    #     sum_spikes = np.repeat(sum_spikes, param.bin_size)
    # else:
    #     sum_spikes = np.sum(binary_spikes, axis=0)
    sum_spikes = np.sum(binary_spikes, axis=0)
    x_value = np.arange(len(spike_nums[0, :]))

    # sp = UnivariateSpline(x_value, sum_spikes, s=240)
    # ax2.fill_between(x_value, 0, smooth_curve(sum_spikes), facecolor="black") # smooth_curve(sum_spikes)
    ax2.fill_between(x_value, 0, sum_spikes, facecolor="black")

    # ax2.yaxis.set_visible(False)
    ax2.set_frame_on(False)
    ax2.get_xaxis().set_visible(True)
    ax2.set_xlim(-1, len(spike_nums[0, :]) + 1)

    if save_raster:
        # if param.with_svg_format:
        #     fig.savefig(f'{param.path_results}/{file_name}_{param.time_str}.svg', format="svg")
        fig.savefig(f'{param.path_results}/{file_name}_{param.time_str}.png')
    # Display the spike raster plot
    if show_raster:
        plt.show()
    plt.close()


def save_data(channels_selection, patient_id, param,
              spike_nums, microwire_labels, cluster_labels,
              spike_nums_ordered, new_index_order, threshold_value):
    np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
             spike_nums=spike_nums, microwire_labels=microwire_labels, spike_nums_ordered=spike_nums_ordered,
             new_index_order=new_index_order, threshold_value=threshold_value,
             cluster_labels=cluster_labels)

def create_spike_train_neo_format(spike_struct):
    spike_trains = []
    t_start = None
    t_stop = None
    for cell in np.arange(len(spike_struct.spike_trains)):
        spike_train = spike_struct.spike_trains[cell]
        # convert frames in ms
        spike_train = spike_train / 1000
        spike_trains.append(spike_train)
        if t_start is None:
            t_start = spike_train[0]
        else:
            t_start = min(t_start, spike_train[0])
        if t_stop is None:
            t_stop = spike_train[-1]
        else:
            t_stop = max(t_stop, spike_train[-1])

    return spike_trains, t_start, t_stop


def filter_spike_trains(spike_trains, cell_labels, threshold, duration_sec):
    """
    Remove cells that fire the most
    Args:
        spike_trains:
        threshold:

    Returns:

    """
    print(f"n cells before filtering: {len(spike_trains)}: ")
    filtered_spike_trains = []
    filtered_cell_labels = []
    for cell in np.arange(len(spike_trains)):
        spike_train = spike_trains[cell]
        n_spike_normalized = len(spike_train) / duration_sec
        # print(f"n spikes: {n_spike_normalized}")
        if n_spike_normalized <= threshold:
            filtered_spike_trains.append(spike_train)
            filtered_cell_labels.append(cell_labels[cell])

    print(f"n cells after filtering: {len(filtered_spike_trains)}")
    return filtered_spike_trains, filtered_cell_labels


def k_mean_clustering(stage_descr, param, path_results_raw, spike_struct, patient,
                      do_filter_spike_trains,
                      n_surrogate_activity_threshold, perc_threshold, debug_mode):
    # ------------------------ params ------------------------
    with_cells_in_cluster_seq_sorted = False
    do_fca_clustering = False
    # kmean clustering
    range_n_clusters_k_mean = np.arange(3, 5)
    n_surrogate_k_mean = 20
    keep_only_the_best_kmean_cluster = False
    # shuffling is necessary to select the significant clusters
    with_shuffling = True

    binsize = 25 * pq.ms

    patient_id = patient.patient_id

    # first we create a spike_trains in the neo format
    spike_trains, t_start, t_stop = create_spike_train_neo_format(spike_struct)

    duration_sec = (t_stop - t_start) / 1000
    print("")
    print(f"## duration in sec {np.round(duration_sec, 3)}")
    print("")
    if duration_sec > 330:
        print(f"skipping this stage because duration > 180 sec")
        path_results = os.path.join(path_results_raw, f"k_mean_{patient_id}_{stage_descr}_skipped")
        os.mkdir(path_results)
        return

    path_results = os.path.join(path_results_raw, f"k_mean_{patient_id}_{stage_descr}")
    os.mkdir(path_results)
    param.path_results = path_results

    cell_labels = spike_struct.labels

    if do_filter_spike_trains:
        filtered_spike_trains, filtered_cell_labels = filter_spike_trains(spike_trains,
                                                                          cell_labels, threshold=5,
                                                                          duration_sec=duration_sec)
        spike_trains = filtered_spike_trains
        cell_labels = filtered_cell_labels
    n_cells = len(spike_trains)

    print(f"Nb units: {len(spike_trains)}")
    for i, train in enumerate(spike_trains):
        print(f"{cell_labels[i]}, nb spikes: {train.shape[0]}")

    neo_spike_trains = []
    for cell in np.arange(n_cells):
        spike_train = spike_trains[cell]
        # print(f"n_spikes: {cell_labels[cell]}: {len(spike_train)}")
        neo_spike_train = neo.SpikeTrain(times=spike_train, units='ms',
                                         t_start=t_start,
                                         t_stop=t_stop)
        neo_spike_trains.append(neo_spike_train)

    spike_trains_binned = elephant_conv.BinnedSpikeTrain(neo_spike_trains, binsize=binsize)

    # transform the binned spiketrain into array
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
        # raise Exception("TEST")
    else:
        spike_nums = spike_trains_binned.to_bool_array().astype("int8")

    print(f"n bins {spike_nums.shape[1]}")
    sliding_window_duration = 1

    activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums,

                                                     window_duration=sliding_window_duration,
                                                     use_max_of_each_surrogate=False,
                                                     spike_train_mode=False,
                                                     n_surrogate=n_surrogate_activity_threshold,
                                                     perc_threshold=perc_threshold,
                                                     debug_mode=False)
    # activity_threshold = get_sce_detection_threshold(spike_nums=spike_struct.spike_trains,
    #
    #                                                  window_duration=sliding_window_duration,
    #                                                  use_max_of_each_surrogate=False,
    #                                                  spike_train_mode=True,
    #                                                  n_surrogate=n_surrogate_activity_threshold,
    #                                                  perc_threshold=perc_threshold,
    #                                                  debug_mode=False)
    print(f"activity_threshold {activity_threshold}")
    if activity_threshold < 2:
        activity_threshold = 2
        print(f"activity_threshold increased to: {activity_threshold}")
    print(f"sliding_window_duration {sliding_window_duration}")
    spike_struct.activity_threshold = activity_threshold
    param.activity_threshold = activity_threshold
    #
    # print("plot_spikes_raster")

    # plot_spikes_raster(spike_nums=spike_struct.spike_trains, param=patient.param,
    #                    spike_train_format=True,
    #                    title=f"raster plot {patient_id}",
    #                    file_name=f"{stage_descr}_test_spike_nums_{patient_id}",
    #                    y_ticks_labels=cell_labels,
    #                    y_ticks_labels_size=4,
    #                    save_raster=True,
    #                    show_raster=False,
    #                    plot_with_amplitude=False,
    #                    activity_threshold=spike_struct.activity_threshold,
    #                    # 500 ms window
    #                    sliding_window_duration=sliding_window_duration,
    #                    show_sum_spikes_as_percentage=True,
    #                    spike_shape="|",
    #                    spike_shape_size=1,
    #                    save_formats="pdf")

    plot_spikes_raster(spike_nums=spike_nums, param=patient.param,
                       spike_train_format=False,
                       title=f"raster plot {patient_id}",
                       file_name=f"{stage_descr}_test_spike_nums_{patient_id}",
                       y_ticks_labels=cell_labels,
                       y_ticks_labels_size=4,
                       save_raster=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       activity_threshold=spike_struct.activity_threshold,
                       # 500 ms window
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       spike_shape="|",
                       spike_shape_size=1,
                       save_formats="pdf")

    # TODO: detect_sce_with_sliding_window with spike_trains
    sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums,
                                                          window_duration=sliding_window_duration,
                                                          perc_threshold=perc_threshold,
                                                          activity_threshold=activity_threshold,
                                                          debug_mode=False)

    print(f"sce_with_sliding_window detected")
    cellsinpeak = sce_detection_result[2]
    SCE_times = sce_detection_result[1]
    sce_times_bool = sce_detection_result[0]
    sce_times_numbers = sce_detection_result[3]
    print(f"Nb SCE: {cellsinpeak.shape}, sce_times len {len(SCE_times)}")
    # raise Exception("GAME OVER")
    # print(f"Nb spikes by SCE: {np.sum(cellsinpeak, axis=0)}")
    # cells_isi = tools_misc.get_isi(spike_data=spike_struct.spike_trains, spike_trains_format=True)
    # for cell_index in np.arange(len(spike_struct.spike_trains)):
    #     print(f"{spike_struct.labels[cell_index]} median isi: {np.round(np.median(cells_isi[cell_index]), 2)}, "
    #           f"mean isi {np.round(np.mean(cells_isi[cell_index]), 2)}")

    # nb_neurons = len(cellsinpeak)

    # return a dict of list of list of neurons, representing the best clusters
    # (as many as nth_best_clusters).
    # the key is the K from the k-mean

    data_descr = f"{patient.patient_id} {stage_descr} sleep"

    if do_fca_clustering:
        n_surrogate_fca = 20
        # sigma=sliding_window_duration*2
        sigma = sliding_window_duration * 0.1

        using_cells_in_peak = True
        if using_cells_in_peak:
            n_cells = len(spike_struct.spike_nums)
            cells_in_peak_trains = []
            cells_in_peak_nums = np.zeros((n_cells, len(SCE_times)), dtype="uint8")
            for cell in np.arange(n_cells):
                cells_spikes = np.where(spike_struct.spike_nums[cell, :])[0]
                spikes_in_sce = cells_spikes[sce_times_numbers[cells_spikes] > -1]
                sce_with_spikes = sce_times_numbers[cells_spikes]
                # removing spikes not in sce
                sce_with_spikes = sce_with_spikes[sce_with_spikes > - 1]
                cells_in_peak_nums[cell, sce_with_spikes] = 1
                cells_in_peak_trains.append(spikes_in_sce)

            compute_and_plot_clusters_raster_fca_version(spike_trains=cells_in_peak_trains,
                                                         spike_nums=spike_struct.spike_nums,
                                                         data_descr=data_descr, param=param,
                                                         sliding_window_duration=sliding_window_duration,
                                                         SCE_times=SCE_times,
                                                         sce_times_numbers=sce_times_numbers,
                                                         perc_threshold=perc_threshold,
                                                         n_surrogate_activity_threshold=
                                                         n_surrogate_activity_threshold,
                                                         sigma=sigma, n_surrogate_fca=n_surrogate_fca,
                                                         labels=cell_labels,
                                                         activity_threshold=activity_threshold,
                                                         fca_early_stop=True,
                                                         use_uniform_jittering=True,
                                                         rolling_surrogate=False,
                                                         with_cells_in_cluster_seq_sorted=with_cells_in_cluster_seq_sorted)

        else:

            compute_and_plot_clusters_raster_fca_version(spike_trains=spike_struct.spike_trains,
                                                         spike_nums=spike_struct.spike_nums,
                                                         data_descr=data_descr, param=param,
                                                         sliding_window_duration=sliding_window_duration,
                                                         SCE_times=SCE_times,
                                                         sce_times_numbers=sce_times_numbers,
                                                         perc_threshold=perc_threshold,
                                                         n_surrogate_activity_threshold=
                                                         n_surrogate_activity_threshold,
                                                         sigma=sigma, n_surrogate_fca=n_surrogate_fca,
                                                         labels=cell_labels,
                                                         activity_threshold=activity_threshold,
                                                         fca_early_stop=True,
                                                         use_uniform_jittering=True,
                                                         rolling_surrogate=True,
                                                         with_cells_in_cluster_seq_sorted=with_cells_in_cluster_seq_sorted)

    else:
        compute_and_plot_clusters_raster_kmean_version(labels=cell_labels,
                                                       activity_threshold=spike_struct.activity_threshold,
                                                       range_n_clusters_k_mean=range_n_clusters_k_mean,
                                                       n_surrogate_k_mean=n_surrogate_k_mean,
                                                       with_shuffling=with_shuffling,
                                                       spike_nums_to_use=spike_nums,
                                                       cellsinpeak=cellsinpeak,
                                                       data_descr=data_descr,
                                                       param=param,
                                                       keep_only_the_best=keep_only_the_best_kmean_cluster,
                                                       sliding_window_duration=sliding_window_duration,
                                                       SCE_times=SCE_times,
                                                       sce_times_numbers=sce_times_numbers,
                                                       perc_threshold=perc_threshold,
                                                       n_surrogate_activity_threshold=
                                                       n_surrogate_activity_threshold,
                                                       debug_mode=debug_mode,
                                                       fct_to_keep_best_silhouettes=np.median,
                                                       with_cells_in_cluster_seq_sorted=with_cells_in_cluster_seq_sorted)

def read_kmean_cell_assembly_file(file_name):
    # list of list, each list correspond to one cell assemblie
    cell_assemblies = []
    # key is the CA index, eachkey is a list correspond to tuples
    # (first and last index of the SCE in frames)
    sce_times_in_single_cell_assemblies = dict()
    sce_times_in_multiple_cell_assemblies = []
    # list of list, each list correspond to tuples (first and last index of the SCE in frames)
    sce_times_in_cell_assemblies = []
    # for each cell, list of list, each correspond to tuples (first and last index of the SCE in frames)
    # in which the cell is supposed to be active for the single cell assemblie to which it belongs
    sce_times_in_cell_assemblies_by_cell = dict()

    with open(file_name, "r", encoding='UTF-8') as file:
        param_section = False
        cell_ass_section = False
        for nb_line, line in enumerate(file):
            if line.startswith("#PARAM#"):
                param_section = True
                continue
            if line.startswith("#CELL_ASSEMBLIES#"):
                cell_ass_section = True
                param_section = False
                continue
            if cell_ass_section:
                if line.startswith("SCA_cluster"):
                    cells = []
                    line_list = line.split(':')
                    cells = line_list[2].split(" ")
                    cell_assemblies.append([int(cell) for cell in cells])
                elif line.startswith("single_sce_in_ca"):
                    line_list = line.split(':')
                    ca_index = int(line_list[1])
                    sce_times_in_single_cell_assemblies[ca_index] = []
                    couples_of_times = line_list[2].split("#")
                    for couple_of_time in couples_of_times:
                        times = couple_of_time.split(" ")
                        sce_times_in_single_cell_assemblies[ca_index].append([int(t) for t in times])
                        sce_times_in_cell_assemblies.append([int(t) for t in times])
                elif line.startswith("multiple_sce_in_ca"):
                    line_list = line.split(':')
                    sces_times = line_list[1].split("#")
                    for sce_time in sces_times:
                        times = sce_time.split(" ")
                        sce_times_in_multiple_cell_assemblies.append([int(t) for t in times])
                        sce_times_in_cell_assemblies.append([int(t) for t in times])
                elif line.startswith("cell"):
                    line_list = line.split(':')
                    if len(line_list) > 3:
                        # it means the cell is not in a SCE
                        continue
                    cell = int(line_list[1])
                    sce_times_in_cell_assemblies_by_cell[cell] = []
                    sces_times = line_list[2].split("#")
                    for sce_time in sces_times:
                        times = sce_time.split()
                        sce_times_in_cell_assemblies_by_cell[cell].append([int(t) for t in times])

    return cell_assemblies, sce_times_in_single_cell_assemblies, sce_times_in_multiple_cell_assemblies, \
           sce_times_in_cell_assemblies, sce_times_in_cell_assemblies_by_cell


def get_stability_among_cell_assemblies(assemblies_1, assemblies_2):
    """
    
    Args:
        assemblies_1: list of int reprensenting the index of a unit
        assemblies_2:

    Returns: list of the same size as assembly_1, of integers representing the percentage of cells in each
    assembly that are part of a same assemby in assemblies_2

    """
    divide_by_total_of_both = True
    perc_list = list()
    for ass_1 in assemblies_1:
        if len(ass_1) == 0:
            continue
        max_perc = 0
        for ass_2 in assemblies_2:
            # easiest way would be to use set() and intersection, but as 2 channels could have the same name
            # we want to have 2 instances different, even so we won't know for sure if that's the same
            n_in_common = len(list(set(ass_1).intersection(ass_2)))
            all_channels = []
            all_channels.extend(ass_1)
            all_channels.extend(ass_2)
            n_different_channels = len(set(all_channels))
            #
            if divide_by_total_of_both:
                perc = (n_in_common / n_different_channels) * 100
            else:
                perc = (n_in_common / len(ass_1)) * 100
            max_perc = max(max_perc, perc)
        perc_list.append(max_perc)
    return perc_list

def read_kmean_results(patients_to_analyse, path_kmean_dir, data_path, path_results, param):
    data_dict = dict()

    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(path_kmean_dir):
        for dir_name in dirnames:
            data_dict[dir_name] = dict()
        break

    if len(data_dict) == 0:
        print("read_kmean_results no directories")
        return

    patients_dict = dict()

    def extract_dir_name_info(dir_name):
        # "k_mean_034fn1_R stage 3 index 10_skipped"
        no_assembly = False
        skipped = False
        index_1st_ = dir_name[2:].find("_") + 2
        index_2nd_ = dir_name[index_1st_+1:].find("_") + index_1st_+1
        patient_id = dir_name[index_1st_+1:index_2nd_]
        other_infos = dir_name[index_2nd_+1:].split()
        recording_side = other_infos[0]
        stage = other_infos[2]
        index_stage = other_infos[4]
        if "skipped" in index_stage:
            index_ = index_stage.find("_")
            index_stage = int(index_stage[:index_])
            skipped = True
        elif "no_assembly" in index_stage:
            index_ = index_stage.find("_")
            index_stage = int(index_stage[:index_])
            no_assembly = True
        else:
            index_stage = int(index_stage)
        return patient_id, recording_side, stage, index_stage, skipped, no_assembly

    info_by_index_dict = dict()
    # first key is the patient
    # value is a dict with 6 keys (differents stages + 'SWS')
    # and value 4 lists representing the n_cells, n_repeat, main_representation and %
    n_cells_repet_dict = dict()
    # first key is the patient id, then the index of the stage,
    for dir_name in data_dict.keys():
        infos = extract_dir_name_info(dir_name)
        # if infos is None:
        #     continue
        patient_id, recording_side, stage, index_stage, skipped, no_assembly = infos

        if patient_id not in patients_to_analyse:
            continue
            
        print(f"patient_id {patient_id}, recording_side {recording_side}, "
              f"stage {stage}, index_stage {index_stage}, skipped {skipped}")
        if skipped or no_assembly:
            continue
        if patient_id not in info_by_index_dict:
            info_by_index_dict[patient_id] = dict()
        if patient_id not in n_cells_repet_dict:
            n_cells_repet_dict[patient_id] = dict()
            for key in ["1", "2", "3", "R", "W", "SWS"]:
                n_cells_repet_dict[patient_id][key] = dict()
                # first one is the total duration and second the number of stages periods in consideration
                # third sum of all the units (SU + MU)
                n_cells_repet_dict[patient_id][key]["count"] = [0, 0, 0]
        if index_stage not in info_by_index_dict[patient_id]:
            info_by_index_dict[patient_id][index_stage] = dict()
        if recording_side not in info_by_index_dict[patient_id][index_stage]:
            info_by_index_dict[patient_id][index_stage][recording_side] = dict()

        info_by_index_dict[patient_id][index_stage][recording_side]["stage"] = stage

        # reading the assembly files in the dire
        assembly_file_name = None
        local_dir = os.path.join(path_kmean_dir, dir_name)
        for (dirpath, dirnames, local_filenames) in os.walk(local_dir):
            for file_name in local_filenames:
                if file_name.endswith(".txt") and "cell_assemblies_data" in file_name:
                    assembly_file_name = file_name
            break
        # print(f"local_dir {local_dir}")
        assembly_file_name = os.path.join(local_dir, assembly_file_name)

        cell_ass_info = read_kmean_cell_assembly_file(assembly_file_name)

        cell_assemblies = cell_ass_info[0]
        sce_times_in_single_cell_assemblies = cell_ass_info[1]
        sce_times_in_multiple_cell_assemblies = cell_ass_info[2]
        sce_times_in_cell_assemblies = cell_ass_info[3]
        sce_times_in_cell_assemblies_by_cell = cell_ass_info[4]

        # loading patient data
        if patient_id not in patients_dict:
            patient = BonnPatient(data_path=data_path, patient_id=patient_id, param=param)
            patients_dict[patient_id] = patient
        else:
            patient = patients_dict[patient_id]

        spike_struct = patient.construct_spike_structure(sleep_stage_indices=[index_stage],
                                                         channels_starting_by=[recording_side],
                                                         spike_trains_format=True,
                                                         spike_nums_format=False,
                                                         keeping_only_SU=False)
        spike_trains, t_start, t_stop = create_spike_train_neo_format(spike_struct)

        duration_sec = (t_stop - t_start) / 1000

        print(f"## duration in sec {np.round(duration_sec, 3)}")


        cell_labels = spike_struct.labels
        do_filter_spike_trains = True
        if do_filter_spike_trains:
            filtered_spike_trains, filtered_cell_labels = filter_spike_trains(spike_trains,
                                                                              cell_labels,
                                                                              threshold=5,
                                                                              duration_sec=duration_sec)
            spike_trains = filtered_spike_trains
            cell_labels = filtered_cell_labels
        n_cells = len(spike_trains)

        # first one is the total duration and second the number of stages periods in consideration
        stages = [stage]
        if stage in '123':
            stages.append("SWS")
        for stage_to_add in stages:
            n_cells_repet_dict[patient_id][stage_to_add]["count"][0] = \
                n_cells_repet_dict[patient_id][stage_to_add]["count"][0] + duration_sec
            n_cells_repet_dict[patient_id][stage_to_add]["count"][1] = \
                n_cells_repet_dict[patient_id][stage_to_add]["count"][1] + 1
            n_cells_repet_dict[patient_id][stage_to_add]["count"][2] = \
                n_cells_repet_dict[patient_id][stage_to_add]["count"][2] + n_cells

        cell_assemblies_by_microwires = list()
        for cell_assembly_index, cell_assembly in enumerate(cell_assemblies):
            n_repeat = 0
            if cell_assembly_index in sce_times_in_single_cell_assemblies:
                n_repeat = len(sce_times_in_single_cell_assemblies[cell_assembly_index])
            print(f"Cell assembly n° {cell_assembly_index}, n repet {n_repeat}")
            cells_str = ""
            microwires_list = []
            for cell in cell_assembly:
                microwires_list.append(cell_labels[cell])
                cells_str = cells_str + f"{cell_labels[cell]} - "
            cells_str = cells_str[:-3]
            cell_assemblies_by_microwires.append(microwires_list)
            counter_dict = count_channels_among_microwires(microwires_list=microwires_list,
                                                           hippocampus_as_one=True)
            print(f"{cells_str}")
            print(f"counter_dict {counter_dict}")
            highest_rate_channel = ""
            highest_rate = 0
            for channel, count in counter_dict.items():
                rate = np.round((count / len(microwires_list)) * 100, 2)
                print(f"{channel}: {rate}")
                if highest_rate < rate:
                    highest_rate = rate
                    highest_rate_channel = channel

            if n_repeat == 0:
                continue
            n_cells_in_ass = len(cell_assembly)
            stages = [stage]
            if stage in '123':
                stages.append("SWS")
            for stage_to_add in stages:
                if highest_rate_channel not in n_cells_repet_dict[patient_id][stage_to_add]:
                    n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel] = [[], [], []]
                n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel][0].append(n_cells_in_ass)
                # n repeat by min
                n_repeat_norm = n_repeat * (60 / duration_sec)
                n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel][1].append(n_repeat_norm)
                n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel][2].append(highest_rate)

        info_by_index_dict[patient_id][index_stage][recording_side][
            "cell_assemblies_mw"] = cell_assemblies_by_microwires
        info_by_index_dict[patient_id][index_stage][recording_side][
            "cell_assemblies_index"] = cell_assemblies
        print("")

    # -------- cell assemblies statbility --------
    same_stage_stabilites = dict()
    # looking 2 stages further
    same_stage_stabilites_gap = dict()
    different_stage_stabilities = dict()
    different_stage_stabilities_gap = dict()
    # now we want to know the proportion of cell assemblies constant from one stage to the other
    for patient_id, index_stage_dict in info_by_index_dict.items():
        if patient_id not in same_stage_stabilites:
            same_stage_stabilites[patient_id] = []
            same_stage_stabilites_gap[patient_id] = []
            different_stage_stabilities[patient_id] = []
            different_stage_stabilities_gap[patient_id] = []
        for index_stage, recording_side_dict in index_stage_dict.items():
            for recording_side, cell_ass_dict in recording_side_dict.items():
                stage = cell_ass_dict["stage"]
                cell_assemblies_by_index = cell_ass_dict["cell_assemblies_index"]
                # getting the one from next stage
                for next_index_stage in [index_stage+1, index_stage+2]:
                    if next_index_stage in index_stage_dict and (recording_side in index_stage_dict[next_index_stage]):
                        next_stage = index_stage_dict[next_index_stage][recording_side]["stage"]
                        next_cell_assemblies = index_stage_dict[next_index_stage][recording_side]["cell_assemblies_index"]
                        stabilities = get_stability_among_cell_assemblies(cell_assemblies_by_index,
                                                                          next_cell_assemblies)
                        same_stages = False
                        if stage in "123":
                            if next_stage in "123":
                                same_stages = True
                        elif stage == "R":
                            if next_stage == "R":
                                same_stages = True
                        elif stage == "W":
                            if next_stage == "W":
                                same_stages = True
                        if same_stages:
                            if next_index_stage == index_stage+1:
                                same_stage_stabilites[patient_id].extend(stabilities)
                            else:
                                same_stage_stabilites_gap[patient_id].extend(stabilities)
                        else:
                            if next_index_stage == index_stage + 1:
                                different_stage_stabilities[patient_id].extend(stabilities)
                            else:
                                different_stage_stabilities_gap[patient_id].extend(stabilities)

    # print(f"same_stage_stabilites {same_stage_stabilites}")
    # print(f"different_stage_stabilities {different_stage_stabilities}")
    # print(f"same_stage_stabilites_gap {same_stage_stabilites_gap}")
    # print(f"different_stage_stabilities_gap {different_stage_stabilities_gap}")
    path_results = os.path.join(path_results, f"cell_ass_analysis_{param.time_str}")
    os.mkdir(path_results)
    for patient_id in same_stage_stabilites.keys():
        box_plot_dict = dict()
        box_plot_dict["same+1"] = same_stage_stabilites[patient_id]
        box_plot_dict["same+2"] = same_stage_stabilites_gap[patient_id]
        box_plot_dict["diff+1"] = different_stage_stabilities[patient_id]
        box_plot_dict["diff+2"] = different_stage_stabilities_gap[patient_id]
        save_formats = ["pdf", "png"]
        # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
        # + 11 diverting

        brewer_colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                         '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                         '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                         '#74add1', '#4575b4', '#313695']
        plot_box_plots(data_dict=box_plot_dict, title="",
                             filename=f"{patient_id}_cell_ass_stabilities",
                             path_results=path_results, with_scatters=True,
                              scatter_size=200,
                             y_label=f"stability (%)", colors=brewer_colors, param=param,
                             save_formats=save_formats)

    #  ---------------- n cells vs n repeat figures ---------------------
    """
    n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel][0].append(n_cells_in_ass)
    n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel][1].append(n_repeat)
    n_cells_repet_dict[patient_id][stage_to_add][highest_rate_channel][2].append(highest_rate)

    """
    for patient_id, stage_dict in n_cells_repet_dict.items():
        for stage, channels_dict in stage_dict.items():
            n_sec = int(channels_dict["count"][0])
            n_stages = channels_dict["count"][1]
            if n_stages == 0:
                n_units = 0
            else:
                n_units = int(channels_dict["count"][2] / n_stages)
            file_name = f"{patient_id}_cells_vs_repeat_cell_assemblies_stage_{stage}_" \
                        f"{n_stages}_periods_{n_sec}_sec_{n_units}_units"
            save_formats = ["pdf", "png"]
            plot_cells_vs_repeat_ass_figure(channels_dict, path_results, file_name, save_formats)


def plot_cells_vs_repeat_ass_figure(channels_dict, path_results, file_name, save_formats):
    """

    Args:
        channels_dict: key channel, value: 3 lists
        path_results:
        file_name:

    Returns:

    """
    background_color = "black"
    labels_color = "white"
    x_labels_rotation = None
    y_log = False
    y_lim = (0, 50)
    x_lim = (0, 35)
    y_label = "n repeat / min"
    x_label = "n cells"

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1]},
                            figsize=(12, 12))

    ax1.set_facecolor(background_color)

    fig.patch.set_facecolor(background_color)

    scatter_size = 400
    markers = ["o", "v", "s", "d", "x"]
    index_channel = 0
    for channel, values in channels_dict.items():
        if channel == "count":
            continue
        x_pos = values[0]
        # adding jitter
        x_pos = [x + ((np.random.random_sample() - 0.5) * 0.25) for x in x_pos]
        y_pos = values[1]
        rates = values[2]
        # from white to red, from low rate to high rate
        brewer_colors = ['#fee5d9','#fcae91','#fb6a4a','#cb181d']
        colors = []
        for rate in rates:
            if rate <= 50:
                colors.append(brewer_colors[0])
            elif rate <= 75:
                colors.append(brewer_colors[1])
            elif rate <= 99:
                colors.append(brewer_colors[2])
            else:
                colors.append(brewer_colors[3])

        ax1.scatter(x_pos, y_pos,
                    color=colors,
                    alpha=1,
                    marker=markers[index_channel],
                    edgecolors=background_color,
                    label=channel,
                    s=scatter_size, zorder=1)
        index_channel += 1

    ax1.legend(labelspacing=2)
    ax1.set_ylabel(f"{y_label}", fontsize=30, labelpad=20)
    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])
    if x_lim is not None:
        ax1.set_xlim(x_lim[0], x_lim[1])
    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=30, labelpad=20)
    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)
    if y_log:
        ax1.set_yscale("log")

    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)

    if x_labels_rotation is not None:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(x_labels_rotation)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)
    fig.tight_layout()
    # adjust the space between axis and the edge of the figure
    # https://matplotlib.org/faq/howto_faq.html#move-the-edge-of-an-axes-to-make-room-for-tick-labels
    # fig.subplots_adjust(left=0.2)

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    for save_format in save_formats:
        fig.savefig(f'{path_results}/{file_name}.{save_format}',
                    format=f"{save_format}",
                    facecolor=fig.get_facecolor())
    plt.close()


def count_channels_among_microwires(microwires_list, hippocampus_as_one):
    """

    Args:
        microwires_list: list of string. Ex: ["MU 51 RA8", "SU 51 RA8"]
        hippocampus_as_one: if True, all parts of hippocampus are considered as one

    Returns:

    """
    # n_microw = len(microwires_list)
    counter_dict = dict()

    for microwire in microwires_list:
        unique_channels = ["EC", "AH", "MH", "PHC"]
        for channel in unique_channels:
            if channel in microwire:
                counter_dict[channel] = counter_dict.get(channel, 0) + 1
        if ("A" in microwire) and ("AH" not in microwire):
            counter_dict["A"] = counter_dict.get("A", 0) + 1
        if ("PH" in microwire) and ("PHC" not in microwire):
            counter_dict["PH"] = counter_dict.get("PH", 0) + 1

    if hippocampus_as_one:
        counter_dict["H"] = 0
        hipp_parts = ["AH", "MH", "PH"]
        for hipp_part in hipp_parts:
            if hipp_part in counter_dict:
                counter_dict["H"] = counter_dict["H"] + counter_dict[hipp_part]
                del counter_dict[hipp_part]
        if counter_dict["H"] == 0:
            del counter_dict["H"]
    return counter_dict

def main():
    root_path = None
    with open("param_bonn.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")

    # root_path=/Users/pappyhammer/Documents/academique/these_inmed/bonn_assemblies/
    data_path = root_path + "one_hour_sleep/"
    path_results_raw = root_path + "results_bonn/"
    k_mean_results_path = os.path.join(root_path, "special_results_bonn",
                                       "kmean_by_stage_25_ms_bin_filtered_SU_and_MU")

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"

    # ------------------------------ param section ------------------------------
    # param will be set later when the spike_nums will have been constructed
    param = BonnParameters(time_str=time_str, path_results=path_results, error_rate=0.2,
                           time_inter_seq=100, min_duration_intra_seq=-5, min_len_seq=10, min_rep_nb=4,
                           max_branches=20, stop_if_twin=False,
                           no_reverse_seq=False, spike_rate_weight=False)

    # --------------------------------------------------------------------------------
    # ------------------------------ param section ------------------------------
    # --------------------------------------------------------------------------------

    patient_ids = ["034fn1", "035fn2", "046fn2", "052fn2"]
    # patient_ids = ["034fn1", "035fn2", "046fn2"]
    # patient_ids = ["046fn2", "052fn2"]
    # patient_ids = ["034fn1"]
    # patient_ids = ["052fn2"]
    # patient_ids = ["046fn2"]
    # patient_ids = ["035fn2"] # memory issue

    decrease_factor = 4  # used to be 4

    sliding_window_duration_in_ms = 25  # 250
    keeping_only_SU = False
    # 100 ms sliding window

    # to find event threshold
    n_surrogate_activity_threshold = 500
    perc_threshold = 95
    # means we remove the units that fire the most
    do_filter_spike_trains = True

    debug_mode = False

    just_do_read_kmean_results = True

    just_do_descriptive_stats = False

    just_do_cad_elephant = False

    # if not just_do_descriptive_stats or not just_do_cad_elephant:
    #     os.mkdir(path_results)

    kwargs_elephant_cad = {"path_results": path_results_raw,
                           "n_cells_min_in_ass_to_plot": 2,
                           "do_filter_spike_trains": False,
                           "alpha_p_value": 0.05,
                           "sliding_window_ms": 25,
                           "time_str": time_str,
                           "with_concatenation": False,
                           "all_sleep_stages_in_order": False}

    # ##########################################################################################
    # #################################### CLUSTERING ###########################################
    # ##########################################################################################
    do_clustering = True

    # ##########################################################################################
    # ################################ PATTERNS SEARCH #########################################
    # ##########################################################################################
    go_for_seq_detection = False
    use_only_uniformity_method = True
    use_loss_score_to_keep_the_best_from_tree = False
    use_sce_times_for_pattern_search = True
    use_ordered_spike_nums_for_surrogate = True
    time_inter_seq_in_ms = 500
    # negative time that can be there between 2 consecutive spikes of a sequence
    min_duration_intra_seq_in_ms = 20
    n_surrogate_for_seq_sorting = 10

    # --------------------------------------------------------------------------------
    # ------------------------------ end param section ------------------------------
    # --------------------------------------------------------------------------------

    if just_do_read_kmean_results:
        read_kmean_results(patients_to_analyse=patient_ids, path_kmean_dir=k_mean_results_path, data_path=data_path, param=param,
                           path_results=path_results_raw)
        return

    for patient_id in patient_ids:
        print(f"patient_id {patient_id}")
        patient = BonnPatient(data_path=data_path, patient_id=patient_id, param=param)
        # patient.print_sleep_stages_info()
        # patient.print_channel_list()

        if just_do_descriptive_stats:
            patient.descriptive_stats()
            continue

        if just_do_cad_elephant:
            patient.elephant_cad(**kwargs_elephant_cad)
            continue

        do_test = False

        if do_test:
            spike_nums_struct = SpikeNumsStructure. \
                load_from_data(path_results_raw +
                               "data_to_load/" + "2nd_SS_left_channels_spike_nums_ordered_046fn2.npz",
                               patient=patient,
                               one_sec=10 ** 6)

            return

        # patient.construct_spike_structure()
        # spike_nums_struct=patient.construct_spike_structure(sleep_stage_indices=[0, 2], sleep_stage_selection=['3'],
        #                              channels_starting_by=["RPH", "RPHC"],
        #                              channels_without_number=["LPH"], channels_with_number=["LA1"])
        # spike_nums_struct = patient.construct_spike_structure(sleep_stage_indices=[2],
        #                              channels_without_number=["RAH"])

        # patient.print_sleep_stages_info(selected_indices=[2])
        build_raster_for_each_stage = False
        if build_raster_for_each_stage:
            # patient.build_raster_for_each_stage_sleep(with_ordering=False, with_concatenation=False,
            #                                           decrease_factor=4,
            #                                           keeping_only_SU=True)
            # patient.build_raster_for_each_stage_sleep(with_ordering=False, with_concatenation=False,
            #                                           decrease_factor=4,
            #                                           keeping_only_SU=False)
            # patient.build_raster_for_each_stage_sleep(with_ordering=False, with_concatenation=True,
            #                                           decrease_factor=4,
            #                                           keeping_only_SU=False)
            patient.build_raster_for_each_stage_sleep(with_ordering=False, with_concatenation=True,
                                                      decrease_factor=4,
                                                      keeping_only_SU=False, all_sleep_stages_in_order=True)
            continue

        # spike_nums_struct = patient.construct_spike_structure(sleep_stage_selection=['2'],
        #                                                  channels_starting_by=["L"],
        #                                                  title="2nd_SS_left_channels")
        # spike_nums_struct = patient.construct_spike_structure(sleep_stage_indices=[2],
        #                                                       channels_starting_by=["L"],
        #                                                       spike_trains_format=False)
        # rem_indices = patient.get_indices_of_sleep_stage(sleep_stage_name='R')
        # stage_2_indices = patient.get_indices_of_sleep_stage(sleep_stage_name='2')
        # print(f"stage_2_indices {stage_2_indices}")
        # raise Exception("toto")

        # put that is inside this for loop in a function
        # for stage_indice in stage_2_indices[2:]:
        for stage_indice in np.arange(len(patient.sleep_stages)):
            if stage_indice != 0:
                continue
            side_to_analyse = "L"
            spike_struct = patient.construct_spike_structure(sleep_stage_indices=[stage_indice],
                                                             channels_starting_by=[side_to_analyse],
                                                             spike_trains_format=True,
                                                             spike_nums_format=False,
                                                             keeping_only_SU=keeping_only_SU)

            # Left_raster_plot_stage_2_046fn2_2018_08_22.21-47-13
            # spike_nums, micro_wires, channels, labels = patient.construct_spike_structure(sleep_stage_indices=[2],
            #                              channels_without_number=["RAH", "RA", "RPHC", "REC", "RMH"])
            # for titles and filenames
            stage_descr = f"{side_to_analyse} stage {patient.sleep_stages[stage_indice].sleep_stage} index {stage_indice}"
            print("")
            print(f"### patient_id {patient_id}: {stage_descr}")
            print("")

            print(f"Nb units: {len(spike_struct.spike_trains)}")
            for i, train in enumerate(spike_struct.spike_trains):
                print(f"{spike_struct.labels[i]}, nb spikes: {train.shape[0]}")

            # spike_struct.decrease_resolution(n=decrease_factor)

            # spike_nums_struct.decrease_resolution (max_time=8)

            ###################################################################
            ###################################################################
            # ###########    SCE detection and clustering        ##############
            ###################################################################
            ###################################################################
            # sliding_window_duration = spike_struct.get_nb_times_by_ms(sliding_window_duration_in_ms,
            #                                                           as_int=True)

            if do_clustering:
                k_mean_clustering(stage_descr, param, path_results_raw, spike_struct, patient,
                                  n_surrogate_activity_threshold, perc_threshold, debug_mode,
                                  do_filter_spike_trains)

            ###################################################################
            ###################################################################
            # ##############    Sequences detection        ###################
            ###################################################################
            ###################################################################

            if not go_for_seq_detection:
                continue

            # ###########    param  ##############
            # around 250 ms
            param.time_inter_seq = spike_struct.get_nb_times_by_ms(time_inter_seq_in_ms,
                                                                   as_int=True)  # 10 ** (6 - decrease_factor) // 4
            param.min_duration_intra_seq = - spike_struct.get_nb_times_by_ms(min_duration_intra_seq_in_ms,
                                                                             as_int=True)
            # -(10 ** (6 - decrease_factor)) // 40
            # a sequence should be composed of at least one third of the neurons
            # param.min_len_seq = len(spike_nums_struct.spike_data) // 4
            param.min_len_seq = 3
            # param.error_rate = param.min_len_seq // 4
            param.error_rate = 0.25
            param.max_branches = 10

            print(f"param.min_len_seq {param.min_len_seq},  param.error_rate {param.error_rate}")

            print(f"spike_nums_struct.activity_threshold {spike_struct.activity_threshold}")

            # continue

            # 2128885
            loss_score = loss_function_with_sliding_window(spike_nums=spike_struct.spike_trains,
                                                           time_inter_seq=param.time_inter_seq,
                                                           spike_train_mode=True,
                                                           min_duration_intra_seq=param.min_duration_intra_seq,
                                                           debug_mode=False)

            print(f'raw loss_score: {np.round(loss_score, 4)}')

            sce_times_bool_to_use = sce_times_bool if use_sce_times_for_pattern_search else None

            find_significant_patterns(spike_nums=spike_struct.spike_nums, param=param,
                                      activity_threshold=activity_threshold,
                                      sliding_window_duration=sliding_window_duration,
                                      data_id=patient.patient_id, n_surrogate=n_surrogate_for_seq_sorting,
                                      extra_file_name=stage_descr, debug_mode=False, without_raw_plot=True,
                                      labels=spike_struct.labels,
                                      sce_times_bool=sce_times_bool_to_use,
                                      use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate,
                                      use_only_uniformity_method=use_only_uniformity_method,
                                      use_loss_score_to_keep_the_best_from_tree=
                                      use_loss_score_to_keep_the_best_from_tree,
                                      spike_shape="o",
                                      spike_shape_size=1,
                                      jitter_links_range=5)

            # spike_struct.spike_data = trains_module.from_spike_trains_to_spike_nums(spike_struct.spike_data)
            #
            # best_seq, seq_dict_real_data = sort_it_and_plot_it(spike_struct=spike_struct, patient=patient, param=param,
            #                                                    channels_selection=stage_descr,
            #                                                    sliding_window_duration=sliding_window_duration,
            #                                                    spike_train_format=False,
            #                                                    sce_times_bool=sce_times_bool_to_use,
            #                                                    debug_mode=True,
            #                                                    use_only_uniformity_method=use_only_uniformity_method,
            #                                                    use_loss_score_to_keep_the_best_from_tree=
            #                                                    use_loss_score_to_keep_the_best_from_tree)
            #
            # nb_cells = len(spike_struct.spike_trains)
            #
            # print("#### REAL DATA ####")
            # print(f"best_seq {best_seq}")
            # real_data_result_for_stat = SortedDict()
            # neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")
            # if seq_dict_real_data is not None:
            #     for key, value in seq_dict_real_data.items():
            #         new_index_key = []
            #         labels_key = []
            #         for i in key:
            #             new_index_key.append(best_seq[i])
            #             labels_key.append(spike_struct.ordered_labels[i])
            #         print(f"len: {len(key)}, new_seq: {key}, labels: {labels_key}, rep: {len(value)}")
            #         if len(key) not in real_data_result_for_stat:
            #             real_data_result_for_stat[len(key)] = []
            #         real_data_result_for_stat[len(key)].append(len(value))
            #         for cell in key:
            #             if neurons_sorted_real_data[cell] == 0:
            #                 neurons_sorted_real_data[cell] = 1
            #
            # n_times = len(spike_struct.spike_nums[0, :])
            # backup_ordered_spike_nums = np.copy(spike_struct.ordered_spike_nums)
            # backup_ordered_labels = spike_struct.ordered_labels[:]
            #
            # print("#### SURROGATE DATA ####")
            # n_surrogate = n_surrogate_for_seq_sorting
            # surrogate_data_result_for_stat = SortedDict()
            # neurons_sorted_surrogate_data = np.zeros(nb_cells, dtype="uint16")
            # for surrogate_number in np.arange(n_surrogate):
            #     copy_spike_nums = np.copy(spike_struct.spike_nums)
            #     for n, neuron_spikes in enumerate(copy_spike_nums):
            #         # roll the data to a random displace number
            #         copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            #     spike_struct.spike_nums = copy_spike_nums
            #
            #     best_seq, seq_dict_surrogate = sort_it_and_plot_it(spike_struct=spike_struct, patient=patient,
            #                                                        param=param,
            #                                                        channels_selection=stage_descr,
            #                                                        title_option=f" surrogate {surrogate_number}",
            #                                                        sliding_window_duration=sliding_window_duration,
            #                                                        spike_train_format=False,
            #                                                        use_only_uniformity_method=use_only_uniformity_method,
            #                                                        use_loss_score_to_keep_the_best_from_tree=
            #                                                        use_loss_score_to_keep_the_best_from_tree)
            #
            #     print(f"best_seq {best_seq}")
            #
            #     mask = np.zeros(nb_cells, dtype="bool")
            #     if seq_dict_surrogate is not None:
            #         for key, value in seq_dict_surrogate.items():
            #             print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
            #             if len(key) not in surrogate_data_result_for_stat:
            #                 surrogate_data_result_for_stat[len(key)] = []
            #             surrogate_data_result_for_stat[len(key)].append(len(value))
            #             for cell in key:
            #                 mask[cell] = True
            #         neurons_sorted_surrogate_data[mask] += 1
            # # min_time, max_time = trains_module.get_range_train_list(spike_nums)
            # # surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
            # #                                               min_value=min_time, max_value=max_time)
            # print("")
            # print("")
            #
            # give_me_stat_on_sorting_seq_results(results_dict=real_data_result_for_stat,
            #                                     neurons_sorted=neurons_sorted_real_data,
            #                                     title="%%%% DATA SET STAT %%%%%", param=param,
            #                                     results_dict_surrogate=surrogate_data_result_for_stat,
            #                                     neurons_sorted_surrogate=neurons_sorted_surrogate_data,
            #                                     n_surrogate=n_surrogate,
            #                                     use_sce_times_for_pattern_search=(sce_times_bool is not None),
            #                                     use_only_uniformity_method=use_only_uniformity_method,
            #                                     use_loss_score_to_keep_the_best_from_tree=
            #                                     use_loss_score_to_keep_the_best_from_tree)
            #
            #
            # significant_threshold_by_seq_len = dict()
            #
            # for key, value in surrogate_data_result_for_stat.items():
            #     significant_threshold_by_seq_len[key] = np.percentile(value, 95)
            #
            # # filtering seq to keep only the significant one
            # significant_seq_dict = dict()
            # for cells, times in seq_dict_real_data.items():
            #     if len(cells) in significant_threshold_by_seq_len:
            #         # print(f"len(cells) {len(cells)}, threshold: {significant_threshold_by_seq_len[key]}")
            #         if len(times) >= significant_threshold_by_seq_len[len(cells)]:
            #             significant_seq_dict[cells] = times
            #     else:
            #         significant_seq_dict[cells] = times
            #
            # title_option = "significant_seq"
            # colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
            #                        "magenta"]
            # plot_spikes_raster(spike_nums=backup_ordered_spike_nums, param=patient.param,
            #                    title=f"raster plot ordered {patient.patient_id} {title_option}",
            #                    spike_train_format=False,
            #                    file_name=f"{stage_descr}_spike_nums_ordered_{patient.patient_id}_{title_option}",
            #                    y_ticks_labels=backup_ordered_labels,
            #                    y_ticks_labels_size=5,
            #                    save_raster=True,
            #                    show_raster=False,
            #                    sliding_window_duration=sliding_window_duration,
            #                    show_sum_spikes_as_percentage=True,
            #                    plot_with_amplitude=False,
            #                    activity_threshold=spike_struct.activity_threshold,
            #                    save_formats="pdf",
            #                    spike_shape="o",
            #                    spike_shape_size=1,
            #                    seq_times_to_color_dict=significant_seq_dict,
            #                    link_seq_color=colors_for_seq_list,
            #                    link_seq_line_width=0.5,
            #                    jitter_links_range=5,
            #                    min_len_links_seq=3)

            return
            # spike_nums_struct.save_data()
    # data_file_dict = dict()
    # data_dict = dict()


def sort_it_and_plot_it(spike_struct, patient, param, channels_selection,
                        sliding_window_duration, title_option="",
                        spike_train_format=False, sce_times_bool=None,
                        debug_mode=False, use_only_uniformity_method=False,
                        use_loss_score_to_keep_the_best_from_tree=False):
    if spike_train_format:
        return
    seq_dict_tmp, \
    best_seq, all_best_seq = order_spike_nums_by_seq(spike_struct.spike_nums, param,
                                                     debug_mode=debug_mode,
                                                     sce_times_bool=sce_times_bool,
                                                     use_only_uniformity_method=use_only_uniformity_method,
                                                     just_keep_the_best=True,
                                                     use_loss_score_to_keep_the_best_from_tree=
                                                     use_loss_score_to_keep_the_best_from_tree)
    # best_seq == corresponding_cells_index
    # if best_seq is None:
    #     print("no sorting order found")
    #     ordered_spike_data = np.copy(spike_struct.spike_nums)
    # else:
    #     ordered_spike_data = np.copy(spike_struct.spike_nums[best_seq, :])
    spike_struct.set_order(ordered_indices=best_seq)
    print(f"starting finding sequences in orderered spike nums")
    seq_dict = find_sequences_in_ordered_spike_nums(spike_nums=spike_struct.ordered_spike_nums, param=param)
    print(f"Sequences in orderered spike nums found")

    # if debug_mode:
    #     print(f"best_seq {best_seq}")
    # if seq_dict_tmp is not None:
    #     colors_for_seq_list = ["blue", "red", "orange", "green", "grey", "yellow", "pink"]
    #     if debug_mode:
    #         for key, value in seq_dict_tmp.items():
    #             print(f"seq: {key}, rep: {len(value)}")
    #
    #     best_seq_mapping_index = dict()
    #     for i, cell in enumerate(best_seq):
    #         best_seq_mapping_index[cell] = i
    #     # we need to replace the index by the corresponding one in best_seq
    #     seq_dict = dict()
    #     for key, value in seq_dict_tmp.items():
    #         new_key = []
    #         for cell in key:
    #             new_key.append(best_seq_mapping_index[cell])
    #         seq_dict[tuple(new_key)] = value
    #
    #     seq_colors = dict()
    #     if debug_mode:
    #         print(f"nb seq to colors: {len(seq_dict)}")
    #     for index, key in enumerate(seq_dict.keys()):
    #         seq_colors[key] = colors_for_seq_list[index % (len(colors_for_seq_list))]
    #         if debug_mode:
    #             print(f"color {seq_colors[key]}, len(seq) {len(key)}")
    # else:
    #     seq_dict = None
    #     seq_colors = None
    # ordered_spike_nums = ordered_spike_data
    # spike_struct.ordered_spike_data = \
    #     trains_module.from_spike_nums_to_spike_trains(spike_struct.ordered_spike_data)

    loss_score = loss_function_with_sliding_window(spike_nums=spike_struct.ordered_spike_nums,
                                                   time_inter_seq=param.time_inter_seq,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   spike_train_mode=False,
                                                   debug_mode=True
                                                   )
    print(f'total loss_score ordered: {np.round(loss_score, 4)}')
    # saving the ordered spike_nums
    # micro_wires_ordered = micro_wires[best_seq]
    # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
    #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

    colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                           "magenta"]
    plot_spikes_raster(spike_nums=spike_struct.ordered_spike_nums, param=patient.param,
                       title=f"raster plot ordered {patient.patient_id} {title_option}",
                       spike_train_format=False,
                       file_name=f"{channels_selection}_spike_nums_ordered_{patient.patient_id}{title_option}",
                       y_ticks_labels=spike_struct.ordered_labels,
                       y_ticks_labels_size=5,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=spike_struct.activity_threshold,
                       save_formats="pdf",
                       spike_shape="o",
                       spike_shape_size=1,
                       seq_times_to_color_dict=seq_dict,
                       link_seq_color=colors_for_seq_list,
                       link_seq_line_width=0.5,
                       jitter_links_range=5,
                       min_len_links_seq=3)

    return best_seq, seq_dict


if __name__ == "__main__":
    main()
