from cicada.analysis.cicada_analysis_format_wrapper import CicadaAnalysisFormatWrapper
import os
import hdf5storage
import numpy as np
from sortedcontainers import SortedList, SortedDict


class SpikeStructure:

    def __init__(self, patient, spike_trains, microwire_labels, cluster_labels, cluster_indices,
                 activity_threshold=None,
                 title=None, ordered_indices=None, ordered_spike_data=None):
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
        self.ordered_spike_data = ordered_spike_data
        # array of int
        self.microwire_labels = np.array(microwire_labels)
        # array of int
        self.cluster_labels = np.array(cluster_labels)
        self.cluster_indices = np.array(cluster_indices)
        self.title = title
        self.labels = self.get_labels()
        self.ordered_indices = ordered_indices
        self.ordered_labels = None
        self.ordered_spike_trains = None
        if self.ordered_indices is not None:
            self.ordered_spike_trains = list()
            for index in ordered_indices:
                self.ordered_spike_trains.append(self.spike_trains[index])
            self.ordered_labels = []
            # y_ticks_labels_ordered = spike_nums_struct.labels[best_seq]
            for old_cell_index in self.ordered_indices:
                self.ordered_labels.append(self.labels[old_cell_index])

    def get_labels(self):
        labels = []
        cluster_to_label = {1: "MU ", 2: "SU ", -1: "Artif ", 0: ""}
        for i, micro_wire in enumerate(self.microwire_labels):
            channel = self.patient.channel_info_by_microwire[micro_wire]
            labels.append(f"{cluster_to_label[self.cluster_labels[i]]}{micro_wire} "
                          f"{channel}")
        return labels

    def set_order(self, ordered_indices):
        if ordered_indices is None:
            self.ordered_spike_trains = np.copy(self.spike_trains)
        else:
            self.ordered_spike_trains = []
            for index in ordered_indices:
                self.ordered_spike_trains.append(self.spike_trains[index])
            self.ordered_indices = ordered_indices
            self.ordered_labels = []
            for old_cell_index in self.ordered_indices:
                self.ordered_labels.append(self.labels[old_cell_index])


class SleepStage:
    def __init__(self, number, start_time, stop_time, sleep_stage, conversion_datetime, conversion_timestamp):
        # timstamps are float, it's needed to multiple by 10^3 to get the real value, with represents microseconds
        self.start_time = start_time * 1000
        self.stop_time = stop_time * 1000
        self.stop_time = stop_time * 1000
        # duration is in microseconds
        self.duration = self.stop_time - self.start_time
        self.duration_sec = self.duration / 1000000
        # string describing the stage (like "W", "3", "R")
        self.sleep_stage = sleep_stage
        self.conversion_datetime = conversion_datetime
        self.conversion_timestamp = conversion_timestamp * 1000
        self.number = number

    def __str__(self):
        result = ""
        result += f"num  {self.number}, "
        result += f"stage  {self.sleep_stage}, "
        # result += f"start_time  {self.start_time}, "
        # result += f"stop_time  {self.stop_time}, \n"
        # result += f"duration (usec)  {self.duration}, "
        result += f"duration: {self.duration_sec:.1f} sec, {(self.duration / 1000000) / 60:.1f} min"
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
        self.available_micro_wires = 0

        self.nb_sleep_stages = 0
        self.available_micro_wires = list()

        if self.load_data_at_init:
            self.load_data()


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
            # return
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
        self.n_microwires = len(self.spikes_by_microwire)
        self.available_micro_wires = np.array(self.available_micro_wires)


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

                print(f"n microwires in {channels_starting_by}: {len(micro_wire_to_keep)}")
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
                print(f"For side {channels_starting_by}: n_su {n_su}, n_mu {n_mu}")
            print(f"mu_by_area_count: {mu_by_area_count}")
            print(f"su_by_area_count: {su_by_area_count}")
            print("")

        print("sleep stages: ")
        for sleep_stage in self.sleep_stages:
            print(sleep_stage)

    @property
    def identifier(self):
        return self._identifier
