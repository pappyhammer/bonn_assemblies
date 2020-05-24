from cicada.analysis.cicada_analysis_format_wrapper import CicadaAnalysisFormatWrapper
import os
import hdf5storage
import numpy as np


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


class CicadaBonnPatient(CicadaAnalysisFormatWrapper):

    DATA_FORMAT = "sEEG_BONN"

    WRAPPER_ID = "BonnPatient"

    def __init__(self, data_ref, load_data=True, verbose=1):
        CicadaAnalysisFormatWrapper.__init__(self, data_ref=data_ref, data_format="sEEG_BONN", load_data=load_data)
        # always load at the start
        self._identifier = os.path.basename(data_ref)
        if verbose:
            print(f"Creating BonnPatient {self._identifier}")
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
        # print(f"sleepstages[0]: {sleepstages[1]}")
        # print(f"conversion_datetime {conversion_datetime}")
        # print(f"conversion_timestamp {conversion_timestamp[0][0]}")
        # print(f"conversion_timestamp int ? {isinstance(conversion_timestamp[0][0], int)}")
        if verbose:
            print(f"total duration (min): {(total_duration / 1000000) / 60}")
        # print_mat_file_content(sleep_stages_file)
        self.available_micro_wires = []
        for file_in_dir in files_in_dir:
            # return
            if file_in_dir.startswith("times_pos_CSC"):
                # -1 to start by 0, to respect other matrices order
                microwire_number = int(file_in_dir[13:-4]) - 1
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
        # everything is loaded in __init__()
        pass

    @property
    def identifier(self):
        return self._identifier
