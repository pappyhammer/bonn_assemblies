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
import pattern_discovery.tools.param as p_disc_param
import pattern_discovery.tools.misc as tools_misc
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.display.raster import plot_dendogram_from_fca
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.trains as trains_module
from pattern_discovery.seq_solver.markov_way import order_spike_nums_by_seq
from pattern_discovery.tools.sce_detection import get_sce_detection_threshold, detect_sce_with_sliding_window
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.clustering.kmean_version.k_mean_clustering import co_var_first_and_clusters
from pattern_discovery.clustering.kmean_version.k_mean_clustering import show_co_var_first_matrix
from pattern_discovery.clustering.fca.fca import functional_clustering_algorithm
import pattern_discovery.clustering.fca.fca as fca


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
        result += f"duration (sec)  {self.duration / 1000000 }, "
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
        self.cluster_info = cluster_info_file["cluster_info"][0, :]
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
            # print(f"sleep_stage_data {sleep_stage_data}")
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
        print(f"total duration (min): {(total_duration/1000000)/60}")
        # print_mat_file_content(sleep_stages_file)
        self.available_micro_wires = []
        for file_in_dir in files_in_dir:
            # return
            if file_in_dir.startswith("times_pos_CSC"):
                # -1 to start by 0, to respect other matrices order
                microwire_number = int(file_in_dir[13:-4]) - 1
                self.available_micro_wires.append(microwire_number)
                data_file = hdf5storage.loadmat(data_path + patient_id + "/" + file_in_dir)
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
                    # timestamp is in microsecond
                    self.spikes_time_by_microwire[microwire_number][index_cluster] = \
                        (cluster_class[mask, 1] * 1000)
                    # .astype(int)
                    # print(f"cluster_class[mask, 1] {cluster_class[mask, 1]}")
                    # print(f"cluster_class[mask, 1] {cluster_class[mask, 1][0] - int(cluster_class[mask, 1][0])}")
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

    def build_raster_for_each_stage_sleep(self, decrease_factor=4,
                                          with_ordering=True,
                                          keeping_only_SU=False, with_concatenation=False):
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
                                                                      keeping_only_SU=keeping_only_SU
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
                    sliding_window_duration = spike_struct.get_nb_times_by_ms(250,
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
                                           y_ticks_labels=spike_struct.labels,
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
        for sleep_stage in ["1", "2", "3", "R", "W"]:
            for side in ["Left", "Right", "Left-Right"]:
                if side == "Left-Right":
                    spike_nums_struct = self.construct_spike_structure(sleep_stage_selection=[sleep_stage],
                                                                       channels_starting_by=["R"])
                else:
                    spike_nums_struct = self.construct_spike_structure(sleep_stage_selection=[sleep_stage],
                                                                       channels_starting_by=[side[0]])

            plot_spikes_raster(spike_nums=spike_nums_struct.spike_nums, param=self.param,
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
        # print(f"micro_wire_to_keep {micro_wire_to_keep}")
        channels_to_keep = [self.channel_info_by_microwire[micro_wire] for micro_wire in micro_wire_to_keep]
        if (sleep_stage_indices is None) and (sleep_stage_selection is None):
            return

        sleep_stages_to_keep = []
        if sleep_stage_indices is not None:
            for index in sleep_stage_indices:
                sleep_stages_to_keep.append(self.sleep_stages[index])

        if sleep_stage_selection is not None:
            sleep_stages_to_keep.extend(self.selection_sleep_stage_by_stage(sleep_stage_selection))

        if len(sleep_stages_to_keep) == 0:
            return None
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


def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/bonn_assemblies/"
    data_path = root_path + "one_hour_sleep/"
    path_results_raw = root_path + "results/"

    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = path_results_raw + f"{time_str}"
    os.mkdir(path_results)

    # ------------------------------ param section ------------------------------
    # param will be set later when the spike_nums will have been constructed
    param = BonnParameters(time_str=time_str, path_results=path_results, error_rate=2,
                           time_inter_seq=100, min_duration_intra_seq=-5, min_len_seq=10, min_rep_nb=4,
                           max_branches=20, stop_if_twin=False,
                           no_reverse_seq=False, spike_rate_weight=False)

    # --------------------------------------------------------------------------------
    # ------------------------------ param section ------------------------------
    # --------------------------------------------------------------------------------

    patient_ids = ["034fn1", "035fn2", "046fn2", "052fn2"]
    patient_ids = ["034fn1"]

    decrease_factor = 4

    sliding_window_duration_in_ms = 250

    # ### sequences paramaters ###
    go_for_seq_detection = False
    time_inter_seq_in_ms = 250
    # negative time that can be there between 2 consecutive spikes of a sequence
    min_duration_intra_seq_in_ms = 20

    # --------------------------------------------------------------------------------
    # ------------------------------ end param section ------------------------------
    # --------------------------------------------------------------------------------

    for patient_id in patient_ids:
        print(f"patient_id {patient_id}")
        patient = BonnPatient(data_path=data_path, patient_id=patient_id, param=param)
        # patient.print_sleep_stages_info()
        # patient.print_channel_list()
        # print(f"param.path_results {patient.param.path_results}")
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
            patient.build_raster_for_each_stage_sleep(with_ordering=False, with_concatenation=False,
                                                      decrease_factor=4,
                                                      keeping_only_SU=True)
            patient.build_raster_for_each_stage_sleep(with_ordering=False, with_concatenation=False,
                                                      decrease_factor=4,
                                                      keeping_only_SU=False)
            continue

        # spike_nums_struct = patient.construct_spike_structure(sleep_stage_selection=['2'],
        #                                                  channels_starting_by=["L"],
        #                                                  title="2nd_SS_left_channels")
        # spike_nums_struct = patient.construct_spike_structure(sleep_stage_indices=[2],
        #                                                       channels_starting_by=["L"],
        #                                                       spike_trains_format=False)
        rem_indices = patient.get_indices_of_sleep_stage(sleep_stage_name='R')
        stage_2_indices = patient.get_indices_of_sleep_stage(sleep_stage_name='2')
        spike_struct = patient.construct_spike_structure(sleep_stage_indices=[stage_2_indices[0]],
                                                         channels_starting_by=["L"],
                                                         spike_trains_format=True,
                                                         spike_nums_format=True,
                                                         keeping_only_SU=True)

        print(f"Nb units: {len(spike_struct.spike_trains)}")
        for i, train in enumerate(spike_struct.spike_trains):
            print(f"{spike_struct.labels[i]}, nb spikes: {train.shape[0]}")

        # Left_raster_plot_stage_2_046fn2_2018_08_22.21-47-13
        # spike_nums, micro_wires, channels, labels = patient.construct_spike_structure(sleep_stage_indices=[2],
        #                              channels_without_number=["RAH", "RA", "RPHC", "REC", "RMH"])
        # for titles and filenames
        channels_selection = "L stage 2"

        spike_struct.decrease_resolution(n=decrease_factor)
        # spike_nums_struct.decrease_resolution (max_time=8)

        ###################################################################
        ###################################################################
        # ###########    SCE detection and clustering        ##############
        ###################################################################
        ###################################################################
        sliding_window_duration = spike_struct.get_nb_times_by_ms(sliding_window_duration_in_ms,
                                                                  as_int=True)

        activity_threshold = get_sce_detection_threshold(spike_nums=spike_struct.spike_trains,
                                                         window_duration=sliding_window_duration,
                                                         spike_train_mode=True,
                                                         n_surrogate=20,
                                                         perc_threshold=99,
                                                         debug_mode=True)
        print(f"activity_threshold {activity_threshold}")
        print(f"sliding_window_duration {sliding_window_duration}")
        spike_struct.activity_threshold = activity_threshold
        param.activity_threshold = activity_threshold

        print("plot_spikes_raster")

        if True:
            plot_spikes_raster(spike_nums=spike_struct.spike_trains, param=patient.param,
                               spike_train_format=True,
                               title=f"raster plot {patient_id}",
                               file_name=f"{channels_selection}_test_spike_nums_{patient_id}",
                               y_ticks_labels=spike_struct.labels,
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
        sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_struct.spike_nums,
                                                              window_duration=sliding_window_duration,
                                                              perc_threshold=95,
                                                              activity_threshold=activity_threshold,
                                                              debug_mode=False)
        print(f"sce_with_sliding_window detected")
        cellsinpeak = sce_detection_result[2]
        SCE_times = sce_detection_result[1]
        print(f"Nb SCE: {cellsinpeak.shape}")
        # print(f"Nb spikes by SCE: {np.sum(cellsinpeak, axis=0)}")
        cells_isi = tools_misc.get_isi(spike_data=spike_struct.spike_trains, spike_trains_format=True)
        for cell_index in np.arange(len(spike_struct.spike_trains)):
            print(f"{spike_struct.labels[cell_index]} median isi: {np.round(np.median(cells_isi[cell_index]), 2)}, "
                  f"mean isi {np.round(np.mean(cells_isi[cell_index]), 2)}")

        nb_neurons = len(cellsinpeak)

        # return a dict of list of list of neurons, representing the best clusters
        # (as many as nth_best_clusters).
        # the key is the K from the k-mean

        data_descr = f"{patient.patient_id} {channels_selection} sleep"

        sarah_way = False
        if sarah_way:
            """
            # sigma=sliding_window_duration*4
            current_cluster = [[[[[[[[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18], 21], 20], 12], 16], 19]]
            merge_history = [[0, 5, 1.1111111111111112], [[0, 5], 1, 1.1111111111111112], [[[0, 5], 1], 2, 1.1111111111111112], [[[[0, 5], 1], 2], 3, 1.1111111111111112], [[[[[0, 5], 1], 2], 3], 4, 1.1111111111111112], [[[[[[0, 5], 1], 2], 3], 4], 6, 1.1111111111111112], [[[[[[[0, 5], 1], 2], 3], 4], 6], 7, 1.1111111111111112], [[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10, 1.1111111111111112], [[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8, 1.1111111111111112], [[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9, 1.1111111111111112], [[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11, 1.1111111111111112], [[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13, 1.1111111111111112], [[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14, 1.1111111111111112], [[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15, 1.1111111111111112], [[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17, 1.1111111111111112], [[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18, 1.0], [[[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18], 21, 1.0], [[[[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18], 21], 20, 0.77777777777777779], [[[[[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18], 21], 20], 12, 0.44444444444444442], [[[[[[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18], 21], 20], 12], 16, 0.22222222222222221], [[[[[[[[[[[[[[[[[[[[[0, 5], 1], 2], 3], 4], 6], 7], 10], 8], 9], 11], 13], 14], 15], 17], 18], 21], 20], 12], 16], 19, 0.0]]

           # sigma=sliding_window_duration/10
           current_cluster =  [[[[[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16], [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12]], 20]]
            merge_history =  [[2, 4, 1.1111111111111112], [0, [2, 4], 1.1111111111111112], [7, 11, 1.1111111111111112], [[7, 11], 13, 1.1111111111111112], [6, [[7, 11], 13], 1.1111111111111112], [[0, [2, 4]], 5, 1.0], [[6, [[7, 11], 13]], 14, 1.0], [[[0, [2, 4]], 5], 1, 0.88888888888888884], [[[6, [[7, 11], 13]], 14], 15, 0.88888888888888884], [[[[0, [2, 4]], 5], 1], 3, 0.33333333333333331], [[[[[0, [2, 4]], 5], 1], 3], 17, 0.66666666666666663], [[[[[[0, [2, 4]], 5], 1], 3], 17], 21, 0.33333333333333331], [[[[6, [[7, 11], 13]], 14], 15], 9, 0.33333333333333331], [[[[[6, [[7, 11], 13]], 14], 15], 9], 10, 0.22222222222222221], [8, 19, 0.22222222222222221], [[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19], 1.0], [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12, 0.44444444444444442], [[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18, 0.1111111111111111], [[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16, 0.22222222222222221], [[[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16], [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12], 0.0], [[[[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16], [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12]], 20, 0.0]]

            """

            compute_and_plot_clusters_raster_sarah_s_way(spike_struct=spike_struct,
                                                         data_descr=data_descr, patient=patient,
                                                         sliding_window_duration=sliding_window_duration,
                                                         SCE_times=SCE_times,
                                                         channels_selection=channels_selection)
        else:
            compute_and_plot_clusters_raster_arnaud_s_way(spike_struct=spike_struct, cellsinpeak=cellsinpeak,
                                                          data_descr=data_descr, patient=patient,
                                                          sliding_window_duration=sliding_window_duration,
                                                          SCE_times=SCE_times,
                                                          channels_selection=channels_selection)

        ###################################################################
        ###################################################################
        # ##############    Sequences detection        ###################
        ###################################################################
        ###################################################################

        if not go_for_seq_detection:
            continue

        # around 250 ms
        param.time_inter_seq = spike_struct.get_nb_times_by_ms(time_inter_seq_in_ms,
                                                               as_int=True)  # 10 ** (6 - decrease_factor) // 4
        param.min_duration_intra_seq = - spike_struct.get_nb_times_by_ms(min_duration_intra_seq_in_ms,
                                                                         as_int=True)
        # -(10 ** (6 - decrease_factor)) // 40
        # a sequence should be composed of at least one third of the neurons
        # param.min_len_seq = len(spike_nums_struct.spike_data) // 4
        param.min_len_seq = 5
        # param.error_rate = param.min_len_seq // 4
        param.error_rate = 0
        param.max_branches = 20

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

        # spike_struct.spike_data = trains_module.from_spike_trains_to_spike_nums(spike_struct.spike_data)

        best_seq, seq_dict = sort_it_and_plot_it(spike_struct=spike_struct, patient=patient, param=param,
                                                 channels_selection=channels_selection,
                                                 sliding_window_duration=sliding_window_duration,
                                                 spike_train_format=False)

        nb_cells = len(spike_struct.spike_trains)

        print("#### REAL DATA ####")
        print(f"best_seq {best_seq}")
        real_data_result_for_stat = SortedDict()
        neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")
        if seq_dict is not None:
            for key, value in seq_dict.items():
                print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
                if len(key) not in real_data_result_for_stat:
                    real_data_result_for_stat[len(key)] = []
                real_data_result_for_stat[len(key)].append(len(value))
                for cell in key:
                    if neurons_sorted_real_data[cell] == 0:
                        neurons_sorted_real_data[cell] = 1

        n_times = len(spike_struct.spike_nums[0, :])

        print("#### SURROGATE DATA ####")
        n_surrogate = 2
        surrogate_data_result_for_stat = SortedDict()
        neurons_sorted_surrogate_data = np.zeros(nb_cells, dtype="uint16")
        for surrogate_number in np.arange(n_surrogate):
            copy_spike_nums = np.copy(spike_struct.spike_nums)
            for n, neuron_spikes in enumerate(copy_spike_nums):
                # roll the data to a random displace number
                copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            spike_struct.spike_nums = copy_spike_nums

            best_seq, seq_dict = sort_it_and_plot_it(spike_struct=spike_struct, patient=patient,
                                                     param=param,
                                                     channels_selection=channels_selection,
                                                     title_option=f" surrogate {surrogate_number}",
                                                     sliding_window_duration=sliding_window_duration,
                                                     spike_train_format=False)

            print(f"best_seq {best_seq}")

            mask = np.zeros(nb_cells, dtype="bool")
            if seq_dict is not None:
                for key, value in seq_dict.items():
                    print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
                    if len(key) not in surrogate_data_result_for_stat:
                        surrogate_data_result_for_stat[len(key)] = []
                    surrogate_data_result_for_stat[len(key)].append(len(value))
                    for cell in key:
                        mask[cell] = True
                neurons_sorted_surrogate_data[mask] += 1
        # min_time, max_time = trains_module.get_range_train_list(spike_nums)
        # surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
        #                                               min_value=min_time, max_value=max_time)
        print("")
        print("")

        give_me_stat_on_sorting_seq_results(results_dict=real_data_result_for_stat,
                                            neurons_sorted=neurons_sorted_real_data,
                                            title="%%%% DATA SET STAT %%%%%", param=param,
                                            results_dict_surrogate=surrogate_data_result_for_stat,
                                            neurons_sorted_surrogate=neurons_sorted_surrogate_data)
        # give_me_stat_on_sorting_seq_results(results_dict=surrogate_data_result_for_stat,
        #                                     neurons_sorted=neurons_sorted_surrogate_data,
        #                                     title="%%%% SURROGATE DATA SET STAT %%%%%", param=param)

        return
        # spike_nums_struct.save_data()
    # data_file_dict = dict()
    # data_dict = dict()


def compute_and_plot_clusters_raster_sarah_s_way(spike_struct, data_descr, patient,
                                                 sliding_window_duration,
                                                 SCE_times, channels_selection):
    param = patient.param
    patient_id = patient.patient_id

    # merge_history, current_cluster = functional_clustering_algorithm(spike_struct.spike_trains, nsurrogate=20,
    #                                                                  sigma=sliding_window_duration / 10,
    #                                                                  early_stop=False,
    #                                                                  rolling_surrogate=False)

    # sigma=sliding_window_duration/2
    current_cluster = [[[[[[[[[[[0, 4], 1], 2], 3], 5], 17], 18], 16],
                         [[[[[[[[[6, [7, 11]], 13], 10], 14], 15], 9], 21], [8, 19]], 12]], 20]]
    merge_history = [[0, 4, 1.1111111111111112], [[0, 4], 1, 1.1111111111111112], [[[0, 4], 1], 2, 1.1111111111111112],
                     [[[[0, 4], 1], 2], 3, 1.1111111111111112], [[[[[0, 4], 1], 2], 3], 5, 1.1111111111111112],
                     [7, 11, 1.1111111111111112], [6, [7, 11], 1.1111111111111112],
                     [[6, [7, 11]], 13, 1.1111111111111112], [[[6, [7, 11]], 13], 10, 1.1111111111111112],
                     [[[[6, [7, 11]], 13], 10], 14, 1.1111111111111112],
                     [[[[[6, [7, 11]], 13], 10], 14], 15, 1.1111111111111112],
                     [[[[[[6, [7, 11]], 13], 10], 14], 15], 9, 1.1111111111111112],
                     [[[[[[0, 4], 1], 2], 3], 5], 17, 0.88888888888888884],
                     [[[[[[[6, [7, 11]], 13], 10], 14], 15], 9], 21, 0.66666666666666663], [8, 19, 0.66666666666666663],
                     [[[[[[[[6, [7, 11]], 13], 10], 14], 15], 9], 21], [8, 19], 1.1111111111111112],
                     [[[[[[[0, 4], 1], 2], 3], 5], 17], 18, 0.55555555555555558],
                     [[[[[[[[0, 4], 1], 2], 3], 5], 17], 18], 16, 0.44444444444444442],
                     [[[[[[[[[6, [7, 11]], 13], 10], 14], 15], 9], 21], [8, 19]], 12, 0.1111111111111111],
                     [[[[[[[[[0, 4], 1], 2], 3], 5], 17], 18], 16],
                      [[[[[[[[[6, [7, 11]], 13], 10], 14], 15], 9], 21], [8, 19]], 12], 0.0], [
                         [[[[[[[[[0, 4], 1], 2], 3], 5], 17], 18], 16],
                          [[[[[[[[[6, [7, 11]], 13], 10], 14], 15], 9], 21], [8, 19]], 12]], 20, 0.0]]
    # # sigma=sliding_window_duration/10
    # current_cluster = [[[[[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16],
    #                      [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12]], 20]]
    # merge_history = [[2, 4, 1.1111111111111112], [0, [2, 4], 1.1111111111111112], [7, 11, 1.1111111111111112],
    #                  [[7, 11], 13, 1.1111111111111112], [6, [[7, 11], 13], 1.1111111111111112], [[0, [2, 4]], 5, 1.0],
    #                  [[6, [[7, 11], 13]], 14, 1.0], [[[0, [2, 4]], 5], 1, 0.88888888888888884],
    #                  [[[6, [[7, 11], 13]], 14], 15, 0.88888888888888884],
    #                  [[[[0, [2, 4]], 5], 1], 3, 0.33333333333333331],
    #                  [[[[[0, [2, 4]], 5], 1], 3], 17, 0.66666666666666663],
    #                  [[[[[[0, [2, 4]], 5], 1], 3], 17], 21, 0.33333333333333331],
    #                  [[[[6, [[7, 11], 13]], 14], 15], 9, 0.33333333333333331],
    #                  [[[[[6, [[7, 11], 13]], 14], 15], 9], 10, 0.22222222222222221], [8, 19, 0.22222222222222221],
    #                  [[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19], 1.0],
    #                  [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12, 0.44444444444444442],
    #                  [[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18, 0.1111111111111111],
    #                  [[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16, 0.22222222222222221],
    #                  [[[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16],
    #                   [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12], 0.0], [
    #                      [[[[[[[[[0, [2, 4]], 5], 1], 3], 17], 21], 18], 16],
    #                       [[[[[[[6, [[7, 11], 13]], 14], 15], 9], 10], [8, 19]], 12]], 20, 0.0]]

    min_scale, max_scale = fca.get_min_max_scale_from_merge_history(merge_history)
    cluster_tree = fca.ClusterTree(clusters_lists=current_cluster[0], merge_history_list=merge_history, father=None,
                                   n_cells=22, max_scale_value=max_scale, non_significant_color="white")

    n_cluster = len(cluster_tree.cluster_nb_list)
    # each index correspond to a cell index, and the value is the cluster the cell belongs,
    # if -1, it means no cluster
    cluster_labels = np.zeros(cluster_tree.n_cells, dtype="int16")
    cluster_labels = cluster_labels - 1
    for cluster in np.arange(n_cluster):
        ct = cluster_tree.cluster_dict[cluster]
        cells_in_cluster = ct.get_cells_id()
        cells_in_cluster = np.array(cells_in_cluster)
        cluster_labels[cells_in_cluster] = cluster

    fig = plt.figure(figsize=(20, 14))
    fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 3})
    outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

    # clusters display
    inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                 subplot_spec=outer[0])

    inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer[1], height_ratios=[10, 2])

    # top is bottom and bottom is top, so the raster is under
    # ax1 contains raster
    ax1 = fig.add_subplot(inner_top[0])

    ax2 = fig.add_subplot(inner_bottom[0])
    # ax3 contains the peak activity diagram
    ax3 = fig.add_subplot(inner_bottom[1], sharex=ax2)

    clustered_spike_nums = np.copy(spike_struct.spike_nums)
    cell_labels = []
    cluster_horizontal_thresholds = []
    cells_to_highlight = []
    cells_to_highlight_colors = []
    start = 0
    for k in np.arange(-1, np.max(cluster_labels) + 1):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        clustered_spike_nums[start:start + nb_k, :] = spike_struct.spike_nums[e, :]
        for index in np.where(e)[0]:
            cell_labels.append(spike_struct.labels[index])
        if k >= 0:
            color = cm.nipy_spectral(float(k + 1) / (n_cluster+1))
            cell_indices = list(np.arange(start, start + nb_k))
            cells_to_highlight.extend(cell_indices)
            cells_to_highlight_colors.extend([color] * len(cell_indices))
        start += nb_k
        if (k + 1) < (np.max(cluster_labels) + 1):
            cluster_horizontal_thresholds.append(start)

    plot_spikes_raster(spike_nums=clustered_spike_nums, param=patient.param,
                       spike_train_format=False,
                       title=f"{n_cluster} clusters raster plot {patient_id}",
                       file_name=f"{channels_selection}_test_spike_nums_{patient_id}_{n_cluster}_clusters",
                       y_ticks_labels=cell_labels,
                       y_ticks_labels_size=4,
                       save_raster=True,
                       show_raster=False,
                       plot_with_amplitude=False,
                       activity_threshold=spike_struct.activity_threshold,
                       span_cells_to_highlight=False,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                       horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                       horizontal_lines_sytle="dashed",
                       vertical_lines=SCE_times,
                       vertical_lines_colors=['white'] * len(SCE_times),
                       vertical_lines_sytle="solid",
                       vertical_lines_linewidth=[0.2] * len(SCE_times),
                       cells_to_highlight=cells_to_highlight,
                       cells_to_highlight_colors=cells_to_highlight_colors,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       spike_shape="o",
                       spike_shape_size=2,
                       save_formats="pdf",
                       axes_list=[ax2, ax3],
                       SCE_times=SCE_times,
                       ylabel="")

    plot_dendogram_from_fca(cluster_tree=cluster_tree, nb_cells=22, save_plot=True, file_name=f"dendogram_{data_descr}",
                            param=param,
                            cell_labels=spike_struct.labels,
                            axes_list=[ax1], fig_to_use=fig)

    plt.close()


def compute_and_plot_clusters_raster_arnaud_s_way(spike_struct, cellsinpeak, data_descr, patient,
                                                  sliding_window_duration,
                                                  SCE_times, channels_selection):
    # -------- clustering params ------ -----
    range_n_clusters_k_mean = np.arange(2, 9)

    param = patient.param
    patient_id = patient.patient_id

    clusters_sce, best_kmeans_by_cluster, m_cov_sces, cluster_labels_for_neurons, surrogate_percentiles = \
        co_var_first_and_clusters(cells_in_sce=cellsinpeak, shuffling=True,
                                  n_surrogate=50,
                                  fct_to_keep_best_silhouettes=np.mean,
                                  range_n_clusters=range_n_clusters_k_mean,
                                  nth_best_clusters=-1,
                                  plot_matrix=False,
                                  data_str=data_descr,
                                  path_results=param.path_results,
                                  neurons_labels=spike_struct.labels)
    do_plot_raster_for_each_clusters = True
    if do_plot_raster_for_each_clusters:
        for n_cluster in range_n_clusters_k_mean:
            clustered_spike_nums = np.copy(spike_struct.spike_nums)
            cell_labels = []
            cluster_labels = cluster_labels_for_neurons[n_cluster]
            cluster_horizontal_thresholds = []
            cells_to_highlight = []
            cells_to_highlight_colors = []
            start = 0
            for k in np.arange(-1, np.max(cluster_labels) + 1):
                e = np.equal(cluster_labels, k)
                nb_k = np.sum(e)
                clustered_spike_nums[start:start + nb_k, :] = spike_struct.spike_nums[e, :]
                for index in np.where(e)[0]:
                    cell_labels.append(spike_struct.labels[index])
                if k >= 0:
                    color = cm.nipy_spectral(float(k + 1) / (n_cluster+1))
                    cell_indices = list(np.arange(start, start + nb_k))
                    cells_to_highlight.extend(cell_indices)
                    cells_to_highlight_colors.extend([color] * len(cell_indices))
                start += nb_k
                if (k + 1) < (np.max(cluster_labels) + 1):
                    cluster_horizontal_thresholds.append(start)
            compile_cluster_result_and_raster = True
            if compile_cluster_result_and_raster:
                fig = plt.figure(figsize=(20, 14))
                fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 2})
                outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

                inner_top = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                             subplot_spec=outer[1], height_ratios=[10, 2])

                # clusters display
                inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 3,
                                                                subplot_spec=outer[0], width_ratios=[6, 10, 6])

                # top is bottom and bottom is top, so the raster is under
                # ax1 contains raster
                ax1 = fig.add_subplot(inner_top[0])
                # ax2 contains the peak activity diagram
                ax2 = fig.add_subplot(inner_top[1], sharex=ax1)

                ax3 = fig.add_subplot(inner_bottom[0])
                # ax2 contains the peak activity diagram
                ax4 = fig.add_subplot(inner_bottom[1])
                ax5 = fig.add_subplot(inner_bottom[2])

                plot_spikes_raster(spike_nums=clustered_spike_nums, param=patient.param,
                                   spike_train_format=False,
                                   title=f"{n_cluster} clusters raster plot {patient_id}",
                                   file_name=f"{channels_selection}_test_spike_nums_{patient_id}_{n_cluster}_clusters",
                                   y_ticks_labels=cell_labels,
                                   y_ticks_labels_size=4,
                                   save_raster=True,
                                   show_raster=False,
                                   plot_with_amplitude=False,
                                   activity_threshold=spike_struct.activity_threshold,
                                   span_cells_to_highlight=False,
                                   raster_face_color='black',
                                   cell_spikes_color='white',
                                   horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                                   horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                                   horizontal_lines_sytle="dashed",
                                   vertical_lines=SCE_times,
                                   vertical_lines_colors=['white'] * len(SCE_times),
                                   vertical_lines_sytle="solid",
                                   vertical_lines_linewidth=[0.2] * len(SCE_times),
                                   cells_to_highlight=cells_to_highlight,
                                   cells_to_highlight_colors=cells_to_highlight_colors,
                                   sliding_window_duration=sliding_window_duration,
                                   show_sum_spikes_as_percentage=True,
                                   spike_shape="o",
                                   spike_shape_size=2,
                                   save_formats="pdf",
                                   axes_list=[ax1, ax2],
                                   SCE_times=SCE_times)

                show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
                                         n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
                                         cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                         data_str=data_descr, path_results=param.path_results,
                                         show_silhouettes=True, neurons_labels=spike_struct.labels,
                                         surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
                                         axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats="pdf")
                plt.close()
            else:
                plot_spikes_raster(spike_nums=clustered_spike_nums, param=patient.param,
                                   spike_train_format=False,
                                   title=f"{n_cluster} clusters raster plot {patient_id}",
                                   file_name=f"{channels_selection}_test_spike_nums_{patient_id}_{n_cluster}_clusters",
                                   y_ticks_labels=cell_labels,
                                   y_ticks_labels_size=4,
                                   save_raster=True,
                                   show_raster=False,
                                   plot_with_amplitude=False,
                                   activity_threshold=spike_struct.activity_threshold,
                                   span_cells_to_highlight=False,
                                   raster_face_color='black',
                                   cell_spikes_color='white',
                                   horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                                   horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                                   horizontal_lines_sytle="dashed",
                                   cells_to_highlight=cells_to_highlight,
                                   cells_to_highlight_colors=cells_to_highlight_colors,
                                   sliding_window_duration=sliding_window_duration,
                                   show_sum_spikes_as_percentage=True,
                                   spike_shape="|",
                                   spike_shape_size=1,
                                   save_formats="pdf")


def give_me_stat_on_sorting_seq_results(results_dict, neurons_sorted, title, param,
                                        results_dict_surrogate=None, neurons_sorted_surrogate=None):
    """
    Key will be the length of the sequence and value will be a list of int, representing the nb of rep
    of the different lists
    :param results_dict:
    :return:
    """
    file_name = f'{param.path_results}/sorting_results_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"{title}" + '\n')
        file.write("" + '\n')
        min_len = 1000
        max_len = 0
        for key in results_dict.keys():
            min_len = np.min((key, min_len))
            max_len = np.max((key, max_len))
        if results_dict_surrogate is not None:
            for key in results_dict_surrogate.keys():
                min_len = np.min((key, min_len))
                max_len = np.max((key, max_len))

        # key reprensents the length of a seq
        for key in np.arange(min_len, max_len + 1):
            nb_seq = None
            nb_seq_surrogate = None
            if key in results_dict:
                nb_seq = results_dict[key]
            if key in results_dict_surrogate:
                nb_seq_surrogate = results_dict_surrogate[key]
            str_to_write = ""
            str_to_write += f"### Length {key}: \n"
            real_data_in = False
            if nb_seq is not None:
                real_data_in = True
                str_to_write += f"# Real data: mean {np.round(np.mean(nb_seq), 3)}"
                if np.std(nb_seq) > 0:
                    str_to_write += f", std {np.round(np.std(nb_seq), 3)}"
            if nb_seq_surrogate is not None:
                if real_data_in:
                    str_to_write += f"\n"
                str_to_write += f"# Surrogate: mean {np.round(np.mean(nb_seq_surrogate), 3)}"
                if np.std(nb_seq_surrogate) > 0:
                    str_to_write += f", std {np.round(np.std(nb_seq_surrogate), 3)}"
            else:
                if not real_data_in:
                    continue
            str_to_write += '\n'
            file.write(f"{str_to_write}")
        file.write("" + '\n')
        file.write("///// Neurons sorted /////" + '\n')
        file.write("" + '\n')

        for index in np.arange(len(neurons_sorted)):
            go_for = False
            if neurons_sorted_surrogate is not None:
                if neurons_sorted_surrogate[index] == 0:
                    pass
                else:
                    go_for = True
            if (not go_for) and neurons_sorted[index] == 0:
                continue
            str_to_write = f"Neuron {index}, x "
            if neurons_sorted_surrogate is not None:
                str_to_write += f"{neurons_sorted_surrogate[index]} / "
            str_to_write += f"{neurons_sorted[index]}"
            if neurons_sorted_surrogate is not None:
                str_to_write += " (surrogate / real data)"
            str_to_write += '\n'
            file.write(f"{str_to_write}")


def sort_it_and_plot_it(spike_struct, patient, param, channels_selection,
                        sliding_window_duration, title_option="",
                        spike_train_format=False,
                        debug_mode=False):
    if spike_train_format:
        return
    seq_dict_tmp, best_seq = order_spike_nums_by_seq(spike_struct.spike_nums, param, with_printing=debug_mode)
    # best_seq == corresponding_cells_index
    # if best_seq is None:
    #     print("no sorting order found")
    #     ordered_spike_data = np.copy(spike_struct.spike_nums)
    # else:
    #     ordered_spike_data = np.copy(spike_struct.spike_nums[best_seq, :])
    spike_struct.set_order(ordered_indices=best_seq)

    if debug_mode:
        print(f"best_seq {best_seq}")
    if seq_dict_tmp is not None:
        colors_for_seq_list = ["blue", "red", "orange", "green", "grey", "yellow", "pink"]
        if debug_mode:
            for key, value in seq_dict_tmp.items():
                print(f"seq: {key}, rep: {len(value)}")

        best_seq_mapping_index = dict()
        for i, cell in enumerate(best_seq):
            best_seq_mapping_index[cell] = i
        # we need to replace the index by the corresponding one in best_seq
        seq_dict = dict()
        for key, value in seq_dict_tmp.items():
            new_key = []
            for cell in key:
                new_key.append(best_seq_mapping_index[cell])
            seq_dict[tuple(new_key)] = value

        seq_colors = dict()
        if debug_mode:
            print(f"nb seq to colors: {len(seq_dict)}")
        for index, key in enumerate(seq_dict.keys()):
            seq_colors[key] = colors_for_seq_list[index % (len(colors_for_seq_list))]
            if debug_mode:
                print(f"color {seq_colors[key]}, len(seq) {len(key)}")
    else:
        seq_dict = None
        seq_colors = None
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
                       save_formats="png",
                       seq_times_to_color_dict=seq_dict,
                       seq_colors=seq_colors)

    return best_seq, seq_dict_tmp


main()
