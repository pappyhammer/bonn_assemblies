"""
Malvache Cell Assemblies Detection
"""

import numpy as np
import neo
import quantities as pq
import os
import elephant.conversion as elephant_conv
import time
import yaml

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

from pymeica.utils.display.rasters import plot_raster
from pymeica.utils.display.colors import BREWER_COLORS

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from sortedcontainers import SortedList, SortedDict

from scipy.special import comb

from pymeica.utils.spike_trains import create_spike_train_neo_format, spike_trains_threshold_by_firing_rate, \
    get_sce_detection_threshold, detect_sce_with_sliding_window

from pymeica.utils.display.rasters import plot_raster


def mcad_main(stage_descr, results_path,
              k_means_cluster_size,
              n_surrogates_k_mean,
              k_mean_n_trials,
              spike_trains_binsize,
              max_size_chunk_in_sec,
              min_size_chunk_in_sec,
              min_activity_threshold,
              spike_nums, spike_trains, cells_label, subject_id,
              n_surrogate_activity_threshold, perc_threshold_for_sce, verbose,
              perc_threshold_for_kmean_surrogates, params_to_save_dict,
              min_n_assemblies=1):
    """

    :param stage_descr:
    :param results_path:
    :param k_means_cluster_size:
    :param n_surrogates_k_mean: int or list of 2 int, indicate how many surrogates should be used to determine if a
    k-mean results pass the threshold. If there are two values, then the first value indicate a low version,
    that is tried fast to save computational time, if it passed the threshold, then the second value is run for
    confirmation.
    :param k_mean_n_trials: int or list of 2 int, indicate how many trial of k-mean should be run to decide which
    cluster holds the best silhouette (the best orthogonality in a sort). It's different than finding a threshold.
    :param min_n_assemblies: if there are two number of surrogates for kmean, this is used to try the second if the
    minimum of assemblies has been found. The assemblies figure is also produce only if this given number is reached
    :param spike_trains_binsize: (int) size of the bin for binning spike train timestamps (in ms)
    :param params_to_save_dict: (dict) params to save in the yaml file, key represent the param id, value its value
    should be simple variable type (list, int, float, str), avoid tuples
    :param spike_nums: 2d binary array (n_cells * n_bins)
    :param spike_trains: list of list (n*cells * n*timestamps)
    :param cells_label:
    :param subject_id:
    :param min_activity_threshold: min number of units to be active to define a syncrhonous event
    :param remove_high_firing_cells:
    :param n_surrogate_activity_threshold:
    :param perc_threshold:
    :param verbose:
    :param firing_rate_threshold:
    :return:
    """

    # ------------------------ params ------------------------
    with_cells_in_cluster_seq_sorted = False
    keep_only_the_best_kmean_cluster = False
    # shuffling is necessary to select the significant clusters
    with_shuffling = True

    results_path = os.path.join(results_path, f"mcad_{subject_id}_{stage_descr}")
    os.mkdir(results_path)

    max_size_chunk_in_bins = (max_size_chunk_in_sec * 1000) // spike_trains_binsize
    min_size_chunk_in_bins = (min_size_chunk_in_sec * 1000) // spike_trains_binsize

    n_cells = len(spike_trains)
    params_to_save_dict["cell_index_to_label"] = dict()
    for unit_index in np.arange(n_cells):
        params_to_save_dict["cell_index_to_label"][int(unit_index)] = str(cells_label[unit_index])

    print(f"Nb units analysed: {len(spike_trains)}")
    print("Below: unit type, cluster, electrode & number of spikes ")
    for i, train in enumerate(spike_trains):
        print(f"{cells_label[i]}, {len(train)}")

    n_bins_in_spike_nums = spike_nums.shape[1]

    # print(f"n bins {n_bins_in_spike_nums}")

    sliding_window_duration = 1

    # -----------------------------------------------------------------------------
    # using chunk_size, we divide spike_nums in several piece if it's too big
    spike_nums_to_process = []
    # list of 2 int representing the first and last bin of the chunk from the whole sleep stage
    first_and_last_bins = []

    if n_bins_in_spike_nums > (max_size_chunk_in_bins + min_size_chunk_in_bins):
        for index_bin in np.arange(0, n_bins_in_spike_nums, max_size_chunk_in_bins):
            if index_bin + max_size_chunk_in_bins > n_bins_in_spike_nums:
                if (n_bins_in_spike_nums - index_bin) < min_size_chunk_in_bins:
                    continue
                spike_nums_chunk = np.copy(spike_nums)[:, index_bin:]
                spike_nums_to_process.append(spike_nums_chunk)
                first_and_last_bins.append([index_bin, n_bins_in_spike_nums - 1])
                continue

            spike_nums_chunk = np.copy(spike_nums)[:, index_bin:index_bin +
                                                                max_size_chunk_in_bins]
            spike_nums_to_process.append(spike_nums_chunk)
            first_and_last_bins.append([index_bin, index_bin + max_size_chunk_in_bins - 1])
    else:
        spike_nums_to_process = [spike_nums]
        first_and_last_bins.append([0, n_bins_in_spike_nums - 1])

    if len(spike_nums_to_process) > 1:
        print(f"Splitting this stage in {len(spike_nums_to_process)} chunks")

    for spike_nums_index, spike_nums in enumerate(spike_nums_to_process):
        first_bin_index, last_bin_index = first_and_last_bins[spike_nums_index]
        if len(spike_nums_to_process) > 1:
            print(f"Processing chunk n° {spike_nums_index} / {len(spike_nums_to_process) - 1} from "
                  f"bin {first_bin_index} to {last_bin_index}")

        bin_descr = f"bins_{first_bin_index}_{last_bin_index}"
        params_to_save_dict.update({"first_bin_index": int(first_bin_index)})
        params_to_save_dict.update({"last_bin_index": int(last_bin_index)})

        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums,
                                                         window_duration=sliding_window_duration,
                                                         use_max_of_each_surrogate=False,
                                                         spike_train_mode=False,
                                                         n_surrogate=n_surrogate_activity_threshold,
                                                         perc_threshold=perc_threshold_for_sce,
                                                         debug_mode=False)

        # we set at 2 the minimum the number of cell to define a synchronous event
        if activity_threshold < min_activity_threshold:
            activity_threshold = min_activity_threshold

        plot_raster(spike_nums=spike_nums, path_results=results_path,
                    spike_train_format=False,
                    title=f"raster plot {subject_id}",
                    file_name=f"{stage_descr}_test_spike_nums_{subject_id}_{bin_descr}",
                    y_ticks_labels=cells_label,
                    y_ticks_labels_size=4,
                    save_raster=True,
                    show_raster=False,
                    plot_with_amplitude=False,
                    activity_threshold=activity_threshold,
                    # 500 ms window
                    sliding_window_duration=sliding_window_duration,
                    show_sum_spikes_as_percentage=True,
                    spike_shape="|",
                    spike_shape_size=1,
                    save_formats="pdf")

        # TODO: detect_sce_with_sliding_window with spike_trains
        sce_detection_result = detect_sce_with_sliding_window(spike_nums=spike_nums,
                                                              window_duration=sliding_window_duration,
                                                              perc_threshold=perc_threshold_for_sce,
                                                              activity_threshold=activity_threshold,
                                                              debug_mode=False)

        # print(f"sce_with_sliding_window detected")
        cellsinpeak = sce_detection_result[2]
        SCE_times = sce_detection_result[1]
        sce_times_bool = sce_detection_result[0]
        sce_times_numbers = sce_detection_result[3]
        print(f"Nb units x SCE: {cellsinpeak.shape}")

        data_descr = f"{subject_id} {stage_descr} sleep {bin_descr}"

        #
        compute_and_plot_clusters_raster_kmean_version(labels=cells_label,
                                                       activity_threshold=activity_threshold,
                                                       range_n_clusters_k_mean=k_means_cluster_size,
                                                       n_surrogate_k_mean=n_surrogates_k_mean,
                                                       k_mean_n_trials=k_mean_n_trials,
                                                       min_n_assemblies=min_n_assemblies,
                                                       with_shuffling=with_shuffling,
                                                       spike_nums_to_use=spike_nums,
                                                       cellsinpeak=cellsinpeak,
                                                       data_descr=data_descr,
                                                       path_results=results_path,
                                                       keep_only_the_best=keep_only_the_best_kmean_cluster,
                                                       sliding_window_duration=sliding_window_duration,
                                                       SCE_times=SCE_times,
                                                       sce_times_numbers=sce_times_numbers,
                                                       perc_threshold_for_sce=perc_threshold_for_sce,
                                                       perc_threshold_for_kmean_surrogates=
                                                       perc_threshold_for_kmean_surrogates,
                                                       n_surrogate_activity_threshold=
                                                       n_surrogate_activity_threshold,
                                                       params_to_save_dict=params_to_save_dict,
                                                       debug_mode=verbose > 0,
                                                       fct_to_keep_best_silhouettes=np.median,
                                                       with_cells_in_cluster_seq_sorted=with_cells_in_cluster_seq_sorted)


class CellAssembly:
    """
    Represent a unique cell assembly. Composed of n cells, for each cell indicate when and how many times
    it repeats. It also give the proportion of Responsive Units...
    """

    def __init__(self, sleep_stage, mcad_outcome, cells, cells_label,
                 is_responsive_units_dict, is_invariant_units_dict,
                 cells_synchronous_event):
        """

        :param sleep_stage: Instance of SleepStage
        :param cells: (list of int) index of the cells
        :param cells_label: list of string, representing cells' label
        :param is_responsive_units_dict: key is the cell label, value is a list of 2 int, representing the evening
        and morning preferred stimulus (-1 means no response)
        :param cells_synchronous_event: (dict) key is the cell index (a value in cells), and the value is a list of
        list of 2 int representing the first and last bin index of each synchronous event in this cell assembly
        """
        self.sleep_stage = sleep_stage
        self.mcad_outcome = mcad_outcome
        self.cells = cells
        self.cells_label = cells_label
        self.is_responsive_units_dict = is_responsive_units_dict
        self.is_invariant_units_dict = is_invariant_units_dict
        self.cells_synchronous_event = cells_synchronous_event
        self._probability_score_pre_computed = None

    @property
    def probability_score(self):
        if self._probability_score_pre_computed is not None:
            # print(f"self._probability_score_pre_computed {self._probability_score_pre_computed}")
            return self._probability_score_pre_computed
        n_responsive_units_total = self.mcad_outcome.n_responsive_units
        n_responsive_units_in_ass = self.n_responsive_units
        cell_ass_prob = (comb(n_responsive_units_total, n_responsive_units_in_ass) *
                         comb((self.mcad_outcome.n_cells_in_total - n_responsive_units_total),
                              (len(self.cells) - n_responsive_units_in_ass))) / comb(self.mcad_outcome.n_cells_in_total,
                                                                                     len(self.cells))
        self._probability_score_pre_computed = cell_ass_prob
        # print(f"cell_ass_prob {cell_ass_prob}")
        return cell_ass_prob

    @property
    def n_units(self):
        return len(self.cells)

    @property
    def n_invariant_units(self):
        return len(self.is_invariant_units_dict)

    @property
    def n_responsive_units(self):
        return len(self.is_responsive_units_dict)

    @property
    def n_repeats(self):
        return len(self.cells_synchronous_event[self.cells[0]])

    #TODO: Compute the probability of this cell assembly taking in consideration the RU & IU

class MCADOutcome:
    BEST_SILHOUETTE = "best_silhouette"

    MAX_N_ASSEMBLIES = "max_n_assemblies"

    def __init__(self, mcad_yaml_dict, comparison_key, subject):
        """
        Allows to easily access outcome of a MCAD that has been computed using the yaml recorded to load the instance.
        :param mcad_yaml_dict: (dict) should be loaded from the yaml file created from the MCAD
        :param comparison_key: (str) should be on the constant of MCADOutcome. Indicate which criteria to take
        in consideration to compare two instances of MCADOutcome.
        :param subject: PyMeicaSubject instance
        """
        # TODO: See to remove cell assemblies with less a certain number of repeat
        #  so far filter is done in load_mcad_data()
        self.subject = subject
        self.mcad_yaml_dict = mcad_yaml_dict
        self.silhouette_score = mcad_yaml_dict["silhouette_score"]
        self.n_cell_assemblies = mcad_yaml_dict["n_cell_assemblies"]

        self.comparison_key = comparison_key
        self.first_bin_index = mcad_yaml_dict["first_bin_index"]
        self.last_bin_index = mcad_yaml_dict["last_bin_index"]
        self.spike_trains_bin_size = mcad_yaml_dict["spike_trains_bin_size"]
        self.sleep_stage_index = mcad_yaml_dict["sleep_stage_index"]
        self.sleep_stage = self.subject.sleep_stages[self.sleep_stage_index]
        self.side = mcad_yaml_dict["side"]
        # only SU, or SU & MU
        self.with_only_SU = mcad_yaml_dict.get("with_only_SU", False)
        # dict with key an int representing the index of the unit in the spike_train that produce this results
        # value is the label with SU cluster channel side region micro_wire
        self.cell_index_to_label = mcad_yaml_dict["cell_index_to_label"]
        # number of cells (units) in total
        self.n_cells_in_total = len(self.cell_index_to_label)
        self.remove_high_firing_cells = "firing_rate_threshold" in mcad_yaml_dict
        self.firing_rate_threshold = mcad_yaml_dict.get("firing_rate_threshold", 0)
        self.cells_synchronous_event = mcad_yaml_dict.get("cells", None)
        # (dict) key is the cell index (a value in cells), and the value is a list of
        #    list of 2 int representing the first and last bin index of each syncrhonous event in this cell assembly
        self.single_cell_assemblies = mcad_yaml_dict.get("single_cell_assemblies", None)

        # index of the list correspond to cell index, and value is a boolean indicating if the cell is IU ou RU
        self.is_cell_responsive_unit = [False] * len(self.cell_index_to_label)
        self.is_cell_invariant_unit = [False] * len(self.cell_index_to_label)
        if len(subject.is_responsive_units_dict) > 0:
            for cell_index, cell_label in self.cell_index_to_label.items():
                if cell_label in subject.is_responsive_units_dict:
                    self.is_cell_responsive_unit[cell_index] = True
        if len(subject.is_invariant_units_dict) > 0:
            for cell_index, cell_label in self.cell_index_to_label.items():
                if cell_label in subject.is_invariant_units_dict:
                    self.is_cell_invariant_unit[cell_index] = True

        self.bins_tuple = (self.first_bin_index, self.last_bin_index)

        if self.n_cell_assemblies > 0:
            # we make sure the cell assemblies are not empty, it happens that actually no syncrhonous event are associated
            # to cell assemblies
            no_cell_assemblies = True
            for cell, list_events in mcad_yaml_dict["cells"].items():
                if len(list_events) > 0:
                    no_cell_assemblies = False
                    break
            if no_cell_assemblies:
                self.n_cell_assemblies = 0

        self.spike_trains, self.spike_nums, \
        cells_label = subject.build_spike_nums(sleep_stage_index=self.sleep_stage_index,
                                               side_to_analyse=self.side,
                                               keeping_only_SU=self.with_only_SU,
                                               remove_high_firing_cells=self.remove_high_firing_cells,
                                               firing_rate_threshold=self.firing_rate_threshold,
                                               spike_trains_binsize=self.spike_trains_bin_size)
        # instance of CellAssembly
        self.cell_assemblies = []
        if (self.n_cell_assemblies > 0) and (self.single_cell_assemblies is not None):
            for cell_assembly_index, cells_in_assembly in self.single_cell_assemblies.items():
                cells_label_in_assembly = [l for index_l, l in enumerate(cells_label) if index_l in cells_in_assembly]
                # self.is_cell_responsive_unit = [False] * len(self.cell_index_to_label)
                is_responsive_units_dict = dict()
                is_invariant_units_dict = dict()
                for cell_label in cells_label_in_assembly:
                    if cell_label in subject.is_responsive_units_dict:
                        is_responsive_units_dict[cell_label] = subject.is_responsive_units_dict[cell_label]
                    if cell_label in subject.is_invariant_units_dict:
                        is_invariant_units_dict[cell_label] = subject.is_invariant_units_dict[cell_label]

                cells_synchronous_event_in_cell_assembly = dict()
                for cell in cells_in_assembly:
                    cells_synchronous_event_in_cell_assembly[cell] = self.cells_synchronous_event[cell]
                cell_assembly = CellAssembly(sleep_stage=self.sleep_stage, mcad_outcome=self,
                                             cells=cells_in_assembly, cells_label=cells_label_in_assembly,
                                             is_responsive_units_dict=is_responsive_units_dict,
                                             is_invariant_units_dict=is_invariant_units_dict,
                                             cells_synchronous_event=cells_synchronous_event_in_cell_assembly)
                self.cell_assemblies.append(cell_assembly)
    # TODO: Make a function that allows to know how many RU / IU in each cell assembly
    #  then how many are stastically

    @property
    def n_responsive_units(self):
        """
        Number of responsive units in this mcad outcome, could be different than subject if RUhas been filtered
        :return:
        """
        return np.sum(self.is_cell_responsive_unit)

    def n_repeats_in_each_cell_assembly(self):
        return [ca.n_repeats for ca in self.cell_assemblies]

    def best_mcad_outcome(self, other_mcad_outcome):
        # first testing if one or the other has no cell assembly
        if self.n_cell_assemblies == 0:
            return other_mcad_outcome
        if other_mcad_outcome.n_cell_assemblies == 0:
            return self

        if self.comparison_key == self.BEST_SILHOUETTE:
            return self._mcad_with_best_silhouette(other_mcad_outcome=other_mcad_outcome)
        if self.comparison_key == self.MAX_N_ASSEMBLIES:
            return self._mcad_with_max_n_assemblies(other_mcad_outcome=other_mcad_outcome)

        return self

    def _mcad_with_max_n_assemblies(self, other_mcad_outcome):
        """
        Return the instance with the best silhouette
        :param other_mcad_outcome:
        :return:
        """
        if self.silhouette_score >= other_mcad_outcome.silhouette_score:
            return self
        return other_mcad_outcome

    def _mcad_with_best_silhouette(self, other_mcad_outcome):
        """
        Return the instance with the max number of assemblies.
        If equal, choose the one with the best silhouette score.
        :param other_mcad_outcome:
        :return:
        """
        if self.n_cell_assemblies > other_mcad_outcome.n_cell_assemblies:
            return self
        if self.n_cell_assemblies < other_mcad_outcome.n_cell_assemblies:
            return other_mcad_outcome

        return self._mcad_with_max_n_assemblies(other_mcad_outcome=other_mcad_outcome)


class CellAssembliesStruct:
    def __init__(self, cellsinpeak, silhouette_score, n_clusters, sce_clusters_id, sce_clusters_labels, neurons_labels,
                 cluster_with_best_silhouette_score, path_results, sliding_window_duration, activity_threshold,
                 n_surrogate_k_mean, SCE_times, data_id):
        """
        Used to help computation of cell assemblies

        :param cellsinpeak:
        :param silhouette_score: Avg silhouette score for this number of cluster
        :param n_clusters:
        :param sce_clusters_id:
        :param sce_clusters_labels:
        :param neurons_labels:
        :param cluster_with_best_silhouette_score:
        :param path_results:
        :param sliding_window_duration:
        :param n_surrogate_k_mean:
        :param SCE_times:
        :param data_id:
        """
        # cellsinpeak shape (n_cells, n_sces)
        self.cellsinpeak = cellsinpeak
        self.path_results = path_results
        self.n_cells = np.shape(cellsinpeak)[0]
        self.n_sces = np.shape(cellsinpeak)[1]
        self.n_clusters = n_clusters
        self.silhouette_score = silhouette_score
        self.n_surrogate_k_mean = n_surrogate_k_mean
        # len(sce_clusters_id) == n_clusters, give the cluster_id for each cluster of sce
        self.sce_clusters_id = sce_clusters_id
        # associate for each cluster_id a sce. len(sce_clusters_labels) == n_sces, and each value is a cluster_id
        self.sce_clusters_labels = sce_clusters_labels
        # give the number of clusters that gives the best silhouette score overall
        self.cluster_with_best_silhouette_score = cluster_with_best_silhouette_score
        self.SCE_times = SCE_times
        self.data_id = data_id

        # to be computed later
        self.cellsinpeak_ordered = None
        # list contains the nb of cells in each cell assemblie cluster
        self.n_cells_in_cell_assemblies_clusters = None
        self.n_cells_not_in_cell_assemblies = None
        # novel order to display cells organized by cell assembly cluster, last cells being the one without cell assembly
        self.cells_indices = None
        # new sce indices
        self.sce_indices = None
        # give the number of sce in the no-assembly-sce, single-assembly and multiple-assembly groups respectively
        self.n_sce_in_assembly = None
        self.n_cells_in_single_cell_assembly_sce_cl = None
        self.n_cells_in_multiple_cell_assembly_sce_cl = None
        self.SCE_times = None
        # threshold used to detect SCE
        self.activity_threshold = activity_threshold
        self.data_descr = None
        # neurons_labels are the label to be display for each cell, in the original order (before any clustering)
        self.neurons_labels = neurons_labels
        self.sliding_window_duration = sliding_window_duration
        # key will be the cell assembly index (such as displayed) and the value is a list of list of 2 elements (first and
        # last sce index of each sce cluster part of this assembly)
        # used to disply lines around clusters in the heatmap
        self.sces_in_cell_assemblies_clusters = None
        # key is a int representing a sce, and value a list representing the id of cell assemblies cluster
        self.cell_assemblies_cluster_of_multiple_ca_sce = None

    def save_data_on_yaml_file(self, n_clusters, params_to_save_dict):
        """
        Save clustering results and params used
        :param n_clusters:
        :param params_to_save_dict:
        :return:
        """
        file_name = os.path.join(self.path_results, f'{self.data_id}_{n_clusters}_mcad.yaml')

        mcad_data_dict = dict()

        # adding params such as for example 'subject_id', bin size...
        mcad_data_dict.update(params_to_save_dict)

        mcad_data_dict['n_clusters'] = int(self.n_clusters)

        mcad_data_dict['n_surrogate_k_mean'] = self.n_surrogate_k_mean
        mcad_data_dict['activity_threshold'] = int(self.activity_threshold)
        mcad_data_dict['silhouette_score'] = float(self.silhouette_score)

        mcad_data_dict['n_cell_assemblies'] = len(self.n_cells_in_cell_assemblies_clusters)

        if len(self.n_cells_in_cell_assemblies_clusters) > 0:
            # single cell assemblies
            mcad_data_dict['single_cell_assemblies'] = dict()
            start = 0
            for cluster_id, n_cells in enumerate(self.n_cells_in_cell_assemblies_clusters):
                stop = start + n_cells
                mcad_data_dict['single_cell_assemblies'][int(cluster_id)] = [int(c) for c in
                                                                             self.cells_indices[start:stop]]
                start = stop

            # MCA_SE: multiple cell-assembly synchronous events
            # for synchronous event (define by the first and last frame indices) we give the id the single
            # cell assemblies that compose it
            for sce_id, ca_ids in self.cell_assemblies_cluster_of_multiple_ca_sce.items():
                if 'multiple_cell_assemblies_synchronous_events' not in mcad_data_dict:
                    mcad_data_dict['multiple_cell_assemblies'] = dict()
                mcad_data_dict['multiple_cell_assemblies'][f"{self.SCE_times[sce_id][0]}-{self.SCE_times[sce_id][1]}"] = \
                    [int(i) for i in ca_ids]

                # then we save the times in bin (first and last of each synchronous event) by cell assemblies
                # first single cell assemblies, then multiple cell assemblies
                n_sces_not_in_ca = self.n_sce_in_assembly[0]
                n_sces_so_far = 0
                start_index = n_sces_not_in_ca + n_sces_so_far
                for ca_id, sces_indices_tuple in self.sces_in_cell_assemblies_clusters.items():
                    n_sces = sces_indices_tuple[0][1] - sces_indices_tuple[0][0]
                    last_index = start_index + n_sces
                    if 'single_se_in_ca' not in mcad_data_dict:
                        mcad_data_dict['single_se_in_ca'] = dict()
                    mcad_data_dict['single_se_in_ca'][int(ca_id)] = []

                    for i, index_sce_period in enumerate(self.sce_indices[start_index:last_index]):
                        sce_period = self.SCE_times[int(index_sce_period)]
                        mcad_data_dict['single_se_in_ca'][int(ca_id)].append([sce_period[0], sce_period[1]])
                    start_index += n_sces

                if len(self.cell_assemblies_cluster_of_multiple_ca_sce) > 0:
                    sce_ids = np.array(list(self.cell_assemblies_cluster_of_multiple_ca_sce.keys()))
                    mcad_data_dict['multiple_se_in_ca'] = []

                    for i, index_sce_period in enumerate(sce_ids):
                        sce_period = self.SCE_times[index_sce_period]
                        mcad_data_dict['multiple_se_in_ca'].append([int(sce_period[0]), int(sce_period[1])])

            # we want to save for each cell at which times it is active in a cell_assembly,
            # if part of a cell assembly
            start = 0
            mcad_data_dict['cells'] = dict()
            for cluster_id, n_cells in enumerate(self.n_cells_in_cell_assemblies_clusters):
                stop = start + n_cells
                for cell in self.cells_indices[start:stop]:
                    n_sces_not_in_ca = self.n_sce_in_assembly[0]
                    n_sces_so_far = 0
                    start_index = n_sces_not_in_ca + n_sces_so_far
                    if cell not in mcad_data_dict['cells']:
                        mcad_data_dict['cells'][int(cell)] = []

                    for ca_id, sces_indices_tuple in self.sces_in_cell_assemblies_clusters.items():
                        n_sces = sces_indices_tuple[0][1] - sces_indices_tuple[0][0]
                        last_index = start_index + n_sces
                        if ca_id != cluster_id:
                            start_index += n_sces
                            continue

                        for i, index_sce_period in enumerate(self.sce_indices[start_index:last_index]):
                            if self.cellsinpeak[cell, index_sce_period] == 0:
                                # print(f"Clustering: Cell {cell} not in sce_index {index_sce_period}")
                                continue
                            sce_period = self.SCE_times[int(index_sce_period)]
                            mcad_data_dict['cells'][int(cell)].append([int(sce_period[0]), int(sce_period[1])])

                        start_index += n_sces

                start = stop

        with open(file_name, 'w') as outfile:
            yaml.dump(mcad_data_dict, outfile, default_flow_style=False, sort_keys=True)

    def save_data_on_file(self, n_clusters):
        file_name = f'{self.path_results}/{self.data_id}_{n_clusters}_clusters_cell_assemblies_data.txt'

        with open(file_name, "w", encoding='UTF-8') as file:
            # first saving params
            file.write(f"#PARAM#" + '\n')
            file.write(f"data_id {self.data_id}" + '\n')
            file.write(f"n_clusters_for_kmean {self.n_clusters}" + '\n')
            file.write(f"n_surrogate_k_mean {self.n_surrogate_k_mean}" + '\n')
            file.write(f"activity_threshold {self.activity_threshold}" + '\n')
            file.write(f"silhouette_score {self.silhouette_score}" + '\n')

            file.write(f"#CELL_ASSEMBLIES#" + '\n')

            if len(self.n_cells_in_cell_assemblies_clusters) == 0:
                file.write(f"NONE" + '\n')
                return

            # SCA_SCE: single cell-assembly SCE
            start = 0
            for cluster_id, n_cells in enumerate(self.n_cells_in_cell_assemblies_clusters):
                stop = start + n_cells
                file.write(f"SCA_cluster:{cluster_id}:{' '.join(map(str, self.cells_indices[start:stop]))}" + '\n')
                start = stop

            # MCA_SCE: multiple cell-assembly SCE
            for sce_id, ca_ids in self.cell_assemblies_cluster_of_multiple_ca_sce.items():
                file.write(f"MCA_SCE:{self.SCE_times[sce_id][0]} {self.SCE_times[sce_id][1]}:")
                file.write(f"{' '.join(map(str, ca_ids))}" + '\n')
            # then if cell assemblies, write the times in frames (first and last of each SCE) by cell assemblies
            # first single cell assemblies, then multiple cell assemblies
            n_sces_not_in_ca = self.n_sce_in_assembly[0]
            n_sces_so_far = 0
            start_index = n_sces_not_in_ca + n_sces_so_far
            for ca_id, sces_indices_tuple in self.sces_in_cell_assemblies_clusters.items():
                n_sces = sces_indices_tuple[0][1] - sces_indices_tuple[0][0]
                last_index = start_index + n_sces
                # print(f"self.sce_indices {self.sce_indices}")
                # print(f"start_index {start_index}")
                # print(f"last_index {last_index}")
                file.write(f"single_sce_in_ca:{ca_id}:")
                for i, index_sce_period in enumerate(self.sce_indices[start_index:last_index]):
                    sce_period = self.SCE_times[int(index_sce_period)]
                    file.write(f"{sce_period[0]} {sce_period[1]}")
                    if i < len(self.sce_indices[start_index:last_index]) - 1:
                        file.write(f"#")
                file.write('\n')
                start_index += n_sces
            if len(self.cell_assemblies_cluster_of_multiple_ca_sce) > 0:
                sce_ids = np.array(list(self.cell_assemblies_cluster_of_multiple_ca_sce.keys()))
                file.write(f"multiple_sce_in_ca:")
                for i, index_sce_period in enumerate(sce_ids):
                    # print(f"index_sce_period {index_sce_period}")
                    # print(f"self.sce_indices[int(index_sce_period)] {self.sce_indices[int(index_sce_period)]}")
                    sce_period = self.SCE_times[index_sce_period]
                    file.write(f"{sce_period[0]} {sce_period[1]}")
                    if i < len(sce_ids) - 1:
                        file.write(f"#")
                file.write('\n')

            # we want to save for each cell at which times it is active in a cell_assembly, if part of a cell assembly
            start = 0
            for cluster_id, n_cells in enumerate(self.n_cells_in_cell_assemblies_clusters):
                stop = start + n_cells
                for cell in self.cells_indices[start:stop]:
                    n_sces_not_in_ca = self.n_sce_in_assembly[0]
                    n_sces_so_far = 0
                    start_index = n_sces_not_in_ca + n_sces_so_far
                    file.write(f"cell:{cell}:")
                    for ca_id, sces_indices_tuple in self.sces_in_cell_assemblies_clusters.items():
                        n_sces = sces_indices_tuple[0][1] - sces_indices_tuple[0][0]
                        last_index = start_index + n_sces
                        if ca_id != cluster_id:
                            start_index += n_sces
                            continue

                        # print(f"self.sce_indices {self.sce_indices}")
                        # print(f"start_index {start_index}")
                        # print(f"last_index {last_index}")
                        to_write = ""
                        for i, index_sce_period in enumerate(self.sce_indices[start_index:last_index]):
                            if self.cellsinpeak[cell, index_sce_period] == 0:
                                # print(f"Clustering: Cell {cell} not in sce_index {index_sce_period}")
                                continue
                            sce_period = self.SCE_times[int(index_sce_period)]
                            to_write = to_write + f"{sce_period[0]} {sce_period[1]}#"
                        if to_write == "":
                            file.write('\n')
                        else:
                            file.write(f'{to_write[:-1]}\n')
                        start_index += n_sces

                start = stop

    def plot_cell_assemblies(self, data_descr, SCE_times, activity_threshold,
                             spike_nums, sce_times_bool=None,
                             display_only_cell_assemblies_on_raster=False,
                             with_cells_in_cluster_seq_sorted=False,
                             save_formats="pdf", show_fig=False):

        self.data_descr = data_descr
        self.SCE_times = SCE_times
        self.activity_threshold = activity_threshold
        background_color = "black"

        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 2})
        outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

        inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                        subplot_spec=outer[1], height_ratios=[10, 2])

        # clusters display
        inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                     subplot_spec=outer[0])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_bottom[0])
        # ax2 contains the peak activity diagram
        ax2 = fig.add_subplot(inner_bottom[1], sharex=ax1)

        ax3 = fig.add_subplot(inner_top[0])
        fig.patch.set_facecolor(background_color)

        self.plot_assemblies_raster(axes_list=[ax1, ax2], spike_nums=spike_nums,
                                    with_cells_in_cluster_seq_sorted=with_cells_in_cluster_seq_sorted,
                                    sce_times_bool=sce_times_bool,
                                    display_only_cell_assemblies_on_raster=display_only_cell_assemblies_on_raster)

        self.plot_cells_vs_sce(ax2=ax3)
        # show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
        #                          significant_sce_clusters=significant_sce_clusters[n_cluster],
        #                          n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
        #                          cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
        #                          data_str=data_descr, path_results=param.path_results,
        #                          show_silhouettes=True, neurons_labels=labels,
        #                          surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
        #                          axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats="pdf")
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        bonus_str = ""
        if display_only_cell_assemblies_on_raster:
            bonus_str = "_only_sca_"
        for save_format in save_formats:
            fig.savefig(f'{self.path_results}/{self.data_descr}_{self.n_clusters}_'
                        f'cell_assemblies{bonus_str}.{save_format}',
                        format=f"{save_format}", facecolor=fig.get_facecolor())
        if show_fig:
            plt.show()
        plt.close()

    def plot_assemblies_raster(self, axes_list, spike_nums, with_cells_in_cluster_seq_sorted=False, sce_times_bool=None,
                               display_only_cell_assemblies_on_raster=False):
        # this section will order the spike_nums for display purpose
        clustered_spike_nums = np.copy(spike_nums)

        cell_labels = []
        for index in self.cells_indices:
            cell_labels.append(self.neurons_labels[index])
        cluster_horizontal_thresholds = []
        cells_to_highlight = []
        cells_to_highlight_colors = []
        start = 0

        clustered_spike_nums = clustered_spike_nums[self.cells_indices, :]

        cells_group_numbers = []
        if len(self.n_cells_in_cell_assemblies_clusters) > 0:
            cells_group_numbers.extend(self.n_cells_in_cell_assemblies_clusters)
        if self.n_cells_not_in_cell_assemblies > 0:
            cells_group_numbers.append(self.n_cells_not_in_cell_assemblies)

        seq_dict = dict()
        colors_for_seq_dict = dict()
        for group_number, group_size in enumerate(cells_group_numbers):
            # if with_cells_in_cluster_seq_sorted and (group_size > 4):
            #     # TODO: use code from markov_way with surrogates generation, need to be more modulable
            #     self.param.error_rate = 0.25
            #     self.param.max_branches = 10
            #     self.param.time_inter_seq = 50
            #     self.param.min_duration_intra_seq = 0
            #     self.param.min_len_seq = 5
            #     self.param.min_rep_nb = 3
            #     # link_seq_categories = significant_category_dict,
            #     # link_seq_color = colors_for_seq_list,
            #     # link_seq_line_width = 0.8,
            #     # link_seq_alpha = 0.9,
            #     to_sort = clustered_spike_nums[start:start + group_size, :]
            #     best_seq, seq_dict_tmp = sort_it_and_plot_it(spike_nums=to_sort, param=self.param,
            #                                                  sce_times_bool=sce_times_bool,
            #                                                  sliding_window_duration=self.sliding_window_duration,
            #                                                  activity_threshold=self.param.activity_threshold,
            #                                                  use_only_uniformity_method=True,
            #                                                  save_plots=False)
            #
            #     # if a list of ordered_indices, the size of the list is equals to ne number of cells,
            #     # each list correspond to the best order with this cell as the first one in the ordered seq
            #     if best_seq is not None:
            #         clustered_spike_nums[start:start + group_size, :] = to_sort[best_seq, :]
            #         color = cm.nipy_spectral(float(group_number + 2) /
            #                                  (len(self.n_cells_in_cell_assemblies_clusters) + 1))
            #         for seq, times in seq_dict_tmp.items():
            #             # updating the cell number to correspond to the whole raster
            #             new_seq = tuple([cell+start for cell in seq])
            #             seq_dict[new_seq] = times
            #             colors_for_seq_dict[new_seq] = color
            #
            #         real_data_result_for_stat = SortedDict()
            #         for key, value in seq_dict_tmp.items():
            #             # print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
            #             if len(key) not in real_data_result_for_stat:
            #                 real_data_result_for_stat[len(key)] = dict()
            #                 real_data_result_for_stat[len(key)]["rep"] = []
            #                 real_data_result_for_stat[len(key)]["duration"] = []
            #             real_data_result_for_stat[len(key)]["rep"].append(len(value))
            #             list_of_durations = []
            #             # keeping the duration of each repetition
            #             for time_stamps in value:
            #                 list_of_durations.append(time_stamps[-1] - time_stamps[0])
            #             real_data_result_for_stat[len(key)]["duration"].append(list_of_durations)
            #         results_dict = real_data_result_for_stat
            #         file_name = f'{self.path_results}/{self.data_descr}_' \
            #                     f'{self.n_clusters}_{group_number}_cell_assemblies_sorting_results_{self.param.time_str}.txt'
            #
            #         min_len = 1000
            #         max_len = 0
            #         with open(file_name, "w", encoding='UTF-8') as file:
            #             for key in results_dict.keys():
            #                 min_len = np.min((key, min_len))
            #                 max_len = np.max((key, max_len))
            #
            #             # key reprensents the length of a seq
            #             for key in np.arange(min_len, max_len + 1):
            #                 nb_rep_seq = None
            #                 flat_durations = None
            #                 if key in results_dict:
            #                     nb_rep_seq = results_dict[key]["rep"]
            #                     durations = results_dict[key]["duration"]
            #                     flat_durations = [item for sublist in durations for item in sublist]
            #
            #                 str_to_write = ""
            #                 str_to_write += f"### Length: {key} cells \n"
            #                 if (nb_rep_seq is not None) and (len(nb_rep_seq) > 0):
            #
            #                     str_to_write += f"# Real data (nb seq: {len(nb_rep_seq)}), " \
            #                                     f"repetition: mean {np.round(np.mean(nb_rep_seq), 3)}"
            #                     if np.std(nb_rep_seq) > 0:
            #                         str_to_write += f", std {np.round(np.std(nb_rep_seq), 3)}"
            #                     str_to_write += f"#, duration: " \
            #                                     f": mean {np.round(np.mean(flat_durations), 3)}"
            #                     if np.std(flat_durations) > 0:
            #                         str_to_write += f", std {np.round(np.std(flat_durations), 3)}"
            #                     str_to_write += f"\n"
            #                 str_to_write += '\n'
            #                 str_to_write += '\n'
            #                 file.write(f"{str_to_write}")

            if (self.n_cells_not_in_cell_assemblies > 0) and (group_number == (len(cells_group_numbers) - 1)):
                continue
            if len(self.n_cells_in_cell_assemblies_clusters) == 0:
                continue

            # coloring cell assemblies
            # color = plt.nipy_spectral(float(group_number + 1) / (len(self.n_cells_in_cell_assemblies_clusters) + 1))
            color = BREWER_COLORS[group_number % len(BREWER_COLORS)]
            # if group_number == 0:
            #     color = "#64D7F7"
            # else:
            #     color = "#D526D7"
            cell_indices_to_color = list(np.arange(start, start + group_size))
            cells_to_highlight.extend(cell_indices_to_color)
            cells_to_highlight_colors.extend([color] * len(cell_indices_to_color))

            start += group_size

            if group_number < (len(cells_group_numbers) - 1):
                cluster_horizontal_thresholds.append(start)

        if len(cell_labels) > 100:
            y_ticks_labels_size = 1
        else:
            y_ticks_labels_size = 3
        spike_shape_size = 1
        if len(cell_labels) > 150:
            spike_shape_size = 0.5
        if len(cell_labels) > 500:
            spike_shape_size = 0.2

        seq_times_to_color_dict = None
        if with_cells_in_cluster_seq_sorted and len(seq_dict) > 0:
            seq_times_to_color_dict = seq_dict

        if display_only_cell_assemblies_on_raster:
            n_cells = len(clustered_spike_nums)
            if (self.n_cells_not_in_cell_assemblies > 0) and (self.n_cells_not_in_cell_assemblies < n_cells):
                n_cells_to_keep = n_cells - int(self.n_cells_not_in_cell_assemblies)
                clustered_spike_nums = clustered_spike_nums[:n_cells_to_keep]
                cell_labels = cell_labels[:n_cells_to_keep]

        colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                               "magenta"]
        # print(f"clustered_spike_nums.shape {clustered_spike_nums.shape}")
        plot_raster(spike_nums=clustered_spike_nums, path_results=self.path_results,
                    spike_train_format=False,
                    title="",
                    file_name=f"cell assemblies raster plot_{self.data_descr}_{self.n_clusters}_clusters",
                    y_ticks_labels=cell_labels,
                    y_ticks_labels_size=y_ticks_labels_size,
                    y_ticks_labels_color="white",
                    x_ticks_labels_color="white",
                    activity_sum_plot_color="white",
                    activity_sum_face_color="black",
                    without_ticks=True,
                    save_raster=False,
                    show_raster=False,
                    plot_with_amplitude=False,
                    activity_threshold=self.activity_threshold,
                    raster_face_color='black',
                    cell_spikes_color='white',
                    horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                    horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                    horizontal_lines_sytle="dashed",
                    horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                    span_area_coords=[self.SCE_times],
                    span_area_colors=['white'],
                    cells_to_highlight=cells_to_highlight,
                    cells_to_highlight_colors=cells_to_highlight_colors,
                    seq_times_to_color_dict=seq_times_to_color_dict,
                    link_seq_color=colors_for_seq_list,  # colors_for_seq_dict,
                    link_seq_line_width=0.6,
                    link_seq_alpha=0.9,
                    jitter_links_range=5,
                    min_len_links_seq=3,
                    sliding_window_duration=self.sliding_window_duration,
                    show_sum_spikes_as_percentage=True,
                    spike_shape="o",
                    spike_shape_size=spike_shape_size,
                    save_formats="pdf",
                    axes_list=axes_list,
                    SCE_times=self.SCE_times)

    def plot_cells_vs_sce(self, ax2):
        # ############################
        # Heatmap cells vs SCE
        # ############################

        n_cell_assemblies = len(self.n_cells_in_cell_assemblies_clusters)

        cluster_horizontal_thresholds = []
        nb_cells_by_cell_ass_cluster_y_coord = []
        # clusters labels
        nb_cells_by_cell_ass_cluster = []
        start = 0

        cells_group_numbers = []
        if n_cell_assemblies > 0:
            cells_group_numbers.extend(self.n_cells_in_cell_assemblies_clusters)
        if self.n_cells_not_in_cell_assemblies > 0:
            cells_group_numbers.append(self.n_cells_not_in_cell_assemblies)

        for cell_group_id, group_size in enumerate(cells_group_numbers):
            # print(f"cell_group_id {cell_group_id}, group_size {group_size}")
            if (self.n_cells_not_in_cell_assemblies > 0) and (cell_group_id == (len(cells_group_numbers) - 1)):
                nb_cells_by_cell_ass_cluster_y_coord.append((start + self.n_cells) / 2)
                nb_cells_by_cell_ass_cluster.append(self.n_cells_not_in_cell_assemblies)
                continue
            if len(self.n_cells_in_cell_assemblies_clusters) == 0:
                continue

            nb_cells_by_cell_ass_cluster_y_coord.append(start + (group_size / 2))
            nb_cells_by_cell_ass_cluster.append(int(group_size))

            range_group = np.arange(start, start + group_size)
            for cell_id in range_group:
                spikes_index = np.where(self.cellsinpeak_ordered[cell_id, :])[0]
                # print(f"spikes_index {spikes_index}, self.n_sce_in_assembly[0] {self.n_sce_in_assembly[0]}")
                # keeping only spikes that are part of sce belonging to cell assemblies
                spikes_index = spikes_index[spikes_index >= self.n_sce_in_assembly[0]]
                # print(f"spikes_index filtered {spikes_index}")
                if len(spikes_index) > 0:
                    # print("you shall not pass")
                    # K +2 to avoid zero and one, and at the end we will substract 2
                    self.cellsinpeak_ordered[cell_id, spikes_index] = cell_group_id + 2

            start += group_size

            if cell_group_id < (len(cells_group_numbers) - 1):
                cluster_horizontal_thresholds.append(start)

        # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
        # value to one represent the cells spikes without assembly, then number 2 represent the cell assembly 0, etc...
        if np.max(self.cellsinpeak_ordered) <= 1:
            # it means that no sce cluster is significant
            list_color = ['black', 'white']
            bounds = [-0.5, 0.5, 1.5]
        else:
            self.cellsinpeak_ordered = self.cellsinpeak_ordered - 2
            # print(f"self.cellsinpeak_ordered {self.cellsinpeak_ordered}")
            # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
            list_color = ['black', 'white']
            bounds = [-2.5, -1.5, -0.5]
            # bounds = [-1.5, 0.5, 1.5]
            for i in np.arange(n_cell_assemblies):
                color = BREWER_COLORS[i % len(BREWER_COLORS)]
                # color = plt.nipy_spectral(float(i + 1) / (n_cell_assemblies + 1))
                # if i == 0:
                #     color = "#64D7F7"
                # else:
                #     color = "#D526D7"
                list_color.append(color)
                bounds.append(i + 0.5)

        cmap = mpl.colors.ListedColormap(list_color)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # rasterized=True used to remove the grid
        sns.heatmap(self.cellsinpeak_ordered, cbar=False, ax=ax2, cmap=cmap, norm=norm, rasterized=True)

        if self.neurons_labels is not None:
            ordered_neurons_labels = []
            for index in self.cells_indices:
                ordered_neurons_labels.append(self.neurons_labels[index])
            ax2.set_yticks(np.arange(len(ordered_neurons_labels)) + 0.5)
            ax2.set_yticklabels(ordered_neurons_labels)
        else:
            ax2.set_yticks(np.arange(self.n_cells) + 0.5)
            ax2.set_yticklabels(self.cells_indices.astype(int))

        if self.n_cells > 100:
            ax2.yaxis.set_tick_params(labelsize=3)
        elif self.n_cells > 200:
            ax2.yaxis.set_tick_params(labelsize=2)
        elif self.n_cells > 400:
            ax2.yaxis.set_tick_params(labelsize=1)
        else:
            ax2.yaxis.set_tick_params(labelsize=4)

        # creating axis at the top
        # ax_top = ax2.twiny()
        ax_right = ax2.twinx()
        # ax2.set_frame_on(False)

        # ax_top.set_frame_on(False)
        # ax_top.set_xlim((0, self.n_sces))
        # ax_top.set_xticks(cluster_sce_x_ticks_coord)
        # sce clusters labels
        # ax_top.set_xticklabels(significant_sce_clusters)

        # print(f"nb_cells_by_cluster_of_cells_y_coord {nb_cells_by_cluster_of_cells_y_coord} "
        #       f"nb_cells_by_cluster_of_cells {nb_cells_by_cluster_of_cells}")

        # ax_right.set_frame_on(False)

        ax_right.set_ylim((0, self.n_cells))
        ax_right.set_yticks(nb_cells_by_cell_ass_cluster_y_coord)
        ax_right.tick_params(axis='both', which='both', length=0)
        # clusters labels
        ax_right.set_yticklabels(nb_cells_by_cell_ass_cluster, fontweight="bold")
        ax_right.yaxis.set_tick_params(labelsize=14)
        for tick_label in ax_right.get_yticklabels():
            tick_label.set_color("white")

        ax2.set_xticks(np.arange(self.n_sces) + 0.5)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        ax2.tick_params(axis='both', which='both', length=0)
        # sce labels
        ax2.set_xticklabels(self.sce_indices.astype(int))
        if self.n_sces > 100:
            ax2.xaxis.set_tick_params(labelsize=3)
        elif self.n_sces > 200:
            ax2.xaxis.set_tick_params(labelsize=2)
        elif self.n_sces > 400:
            ax2.xaxis.set_tick_params(labelsize=1)
        else:
            ax2.xaxis.set_tick_params(labelsize=4)
        # horizontal lines showing the border between cell assemblies clusters, not so necessary as they are colored
        # ax2.hlines(cluster_horizontal_thresholds, self.n_sce_in_assembly[0], self.n_sces,
        #            color="white", linewidth=1,
        #            linestyles="dashed")
        # put a line between sce with one single cell assembly and those with multiple
        ax2.vlines([self.n_sce_in_assembly[0] + self.n_sce_in_assembly[1]], 0,
                   self.n_cells - self.n_cells_not_in_cell_assemblies, color="white", linewidth=2,
                   linestyles="dashed")
        start = 0
        for index, group_size in enumerate(self.n_cells_in_cell_assemblies_clusters):
            if index not in self.sces_in_cell_assemblies_clusters:
                # then there is no significant sce cluster
                start += group_size
                continue
            sces_clusters_borders = self.sces_in_cell_assemblies_clusters[index]
            for sces_borders in sces_clusters_borders:
                y_bottom = start
                y_top = start + group_size
                x_left = sces_borders[0]
                x_right = sces_borders[1]
                linewidth = 2
                color_border = "white"
                ax2.vlines(x_left, y_bottom, y_top, color=color_border, linewidth=linewidth)
                ax2.vlines(x_right, y_bottom, y_top, color=color_border, linewidth=linewidth)
                ax2.hlines(y_bottom, x_left, x_right, color=color_border, linewidth=linewidth)
                ax2.hlines(y_top, x_left, x_right, color=color_border, linewidth=linewidth)
            start += group_size

        # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)
        ax2.tick_params(axis='y', colors="white")
        ax2.tick_params(axis='x', colors="white")
        start = 0
        for i, cell_group_size in enumerate(self.n_cells_in_cell_assemblies_clusters):
            # color = plt.nipy_spectral(float(i + 1) / (n_cell_assemblies + 1))
            color = BREWER_COLORS[i % len(BREWER_COLORS)]
            # if i == 0:
            #     color = "#64D7F7"
            # else:
            #     color = "#D526D7"
            for index in np.arange(start, (start + cell_group_size)):
                ax2.get_yticklabels()[index].set_color(color)
            start += cell_group_size

        ax2.invert_yaxis()


def load_cell_assemblies_data(data_file_name):
    """
    Load data concerning cell assemblies detection using Malvache et al. method.
    Args:
        data_file_name: File_name (with path) of the data file containing the result of the analysis (txt file)

    Returns: a list with 5 structures
    1. cell_assemblies: list of list, each list correspond to one cell assembly
    2. sce_times_in_single_cell_assemblies: dict, key is the CA index, each value is a list correspond to tuples
    (first and last index of the SCE in frames)
    3. sce_times_in_multiple_cell_assemblies: list of tuples representing the the first and last index of SCE part of
    multiple cell assemblies
    4. sce_times_in_cell_assemblies: list of list, each list correspond to tuples
    (first and last index of the SCE in frames)
    5. sce_times_in_cell_assemblies_by_cell: dict, for each cell, list of list, each correspond to tuples
    (first and last index of the SCE in frames) in which the cell is supposed
    to be active for the single cell assembly to which it belongs

    """
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
    # TODO: Add loading of silhouette score
    with open(data_file_name, "r", encoding='UTF-8') as file:
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
                    cell = int(line_list[1])
                    sce_times_in_cell_assemblies_by_cell[cell] = []
                    sces_times = line_list[2].split("#")
                    for sce_time in sces_times:
                        times = sce_time.split()
                        sce_times_in_cell_assemblies_by_cell[cell].append([int(t) for t in times])

    results = list()
    results.append(cell_assemblies)
    results.append(sce_times_in_single_cell_assemblies)
    results.append(sce_times_in_multiple_cell_assemblies)
    results.append(sce_times_in_cell_assemblies)
    results.append(sce_times_in_cell_assemblies_by_cell)
    return results


def covnorm(m_sces):
    nb_events = np.shape(m_sces)[1]
    co_var_matrix = np.zeros((nb_events, nb_events))
    for i in np.arange(nb_events):
        for j in np.arange(nb_events):
            if np.correlate(m_sces[:, i], m_sces[:, j]) == 0:
                co_var_matrix[i, j] = 0
            else:
                # we remove the mean to do the same as xcov is doing on matlab
                co_var_matrix[i, j] = np.correlate((m_sces[:, i] - np.mean(m_sces[:, i])),
                                                   (m_sces[:, j] - np.mean(m_sces[:, j]))) / np.std(m_sces[:, i]) \
                                      / np.std(m_sces[:, j]) / nb_events
    return co_var_matrix


def surrogate_clustering(m_sces, n_clusters, n_surrogate, n_trials, perc_threshold,
                         fct_to_keep_best_silhouettes, debug_mode=False):
    """

    :param m_sces: sce matrix used for the clustering after permutation
    :param n_clusters: number of clusters
    :param n_surrogate: number of surrogates
    :param n_trials: number of trials by surrogate, keeping one avg silhouette by surrogate
    :param perc_threshold: threshold as percentile (int)
    :param debug_mode:
    :return: a list of value representing the nth percentile over the average threshold of each surrogate, keeping
    each individual silhouette score, not just the mean of each surrogate
    """
    original_m_sces = m_sces
    surrogate_silhouettes = np.zeros(n_surrogate * n_clusters)

    # TODO: Increase the speed of this loop
    # print(f'Start clustering on shuffled data: {n_surrogate} surrogates, {n_trials} trials per surrogate')
    for surrogate_index in np.arange(n_surrogate):
        m_sces = np.copy(original_m_sces)
        for n, sces in enumerate(m_sces):
            m_sces[n, :] = np.roll(sces, np.random.randint(1, m_sces.shape[1]))
        # for j in np.arange(len(m_sces[0])):
        #     m_sces[:, j] = np.random.permutation(m_sces[:, j])
        # for i, s in enumerate(m_sces):
        #     # print(f"pos before permutation {np.where(s)[0]}")
        #     m_sces[i] = np.random.permutation(s)
        best_silhouettes_clusters_avg = np.zeros(n_clusters)
        best_median_silhouettes = 0

        for trial in np.arange(n_trials):
            # co_var = np.cov(m_sces)
            kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
            cluster_labels = kmeans.labels_
            silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
            sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
            local_clusters_silhouette = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values
                # ith_cluster_silhouette_values.sort()
            med = fct_to_keep_best_silhouettes(local_clusters_silhouette)
            if med > best_median_silhouettes:
                best_median_silhouettes = med
                best_silhouettes_clusters_avg = local_clusters_silhouette

        index = surrogate_index * n_clusters
        surrogate_silhouettes[index:index + n_clusters] = best_silhouettes_clusters_avg

    # if debug_mode:
    #     print(f'End of clustering on shuffled data')
    # print(f"surrogate_silhouettes {surrogate_silhouettes}, perc_threshold {perc_threshold}")
    percentile_result = np.percentile(surrogate_silhouettes, perc_threshold)

    return percentile_result


def _apply_kmean_on_msces_with_trials(m_sces_list):
    n_trials = 100
    n_clusters = 3
    fct_to_keep_best_silhouettes = np.mean

    surrogate_silhouettes = np.zeros(0)

    for m_sces in m_sces_list:
        best_silhouettes_clusters_avg = np.zeros(n_clusters)
        best_median_silhouettes = 0
        for trial in np.arange(n_trials):
            # co_var = np.cov(m_sces)
            kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
            cluster_labels = kmeans.labels_
            silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
            sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
            local_clusters_silhouette = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values
                # ith_cluster_silhouette_values.sort()
            med = fct_to_keep_best_silhouettes(local_clusters_silhouette)
            if med > best_median_silhouettes:
                best_median_silhouettes = med
                best_silhouettes_clusters_avg = local_clusters_silhouette
        surrogate_silhouettes = np.hstack((surrogate_silhouettes, best_silhouettes_clusters_avg))
    return surrogate_silhouettes


def surrogate_clustering_multi_process(m_sces, n_clusters, n_surrogate, n_trials, perc_threshold,
                                       fct_to_keep_best_silhouettes, debug_mode=False):
    """

    :param m_sces: sce matrix used for the clustering after permutation
    :param n_clusters: number of clusters
    :param n_surrogate: number of surrogates
    :param n_trials: number of trials by surrogate, keeping one avg silhouette by surrogate
    :param perc_threshold: threshold as percentile (int)
    :param debug_mode:
    :return: a list of value representing the nth percentile over the average threshold of each surrogate, keeping
    each individual silhouette score, not just the mean of each surrogate
    """
    # TODO: Multi-processing slower than non...
    original_m_sces = m_sces
    surrogate_silhouettes = np.zeros(n_surrogate * n_clusters)

    # for j in np.arange(len(m_sces[0])):
    #     m_sces[:, j] = np.random.permutation(m_sces[:, j])
    # for i, s in enumerate(m_sces):
    #     # print(f"pos before permutation {np.where(s)[0]}")
    #     m_sces[i] = np.random.permutation(s)

    cores = mp.cpu_count()
    print(f"N cores in computer: {cores}")

    # create the multiprocessing pool
    pool = Pool(cores)

    # TODO: Increase the speed of this loop
    print(f'Start clustering on shuffled data: {n_surrogate} surrogates, {n_trials} trials per surrogate')
    m_sces_list = []
    for core_n in np.arange(cores):
        sub_m_sces_list = []
        for surrogate_index in np.arange(n_surrogate // cores):
            m_sces = np.copy(original_m_sces)
            for n, sces in enumerate(m_sces):
                m_sces[n, :] = np.roll(sces, np.random.randint(1, m_sces.shape[1]))
            sub_m_sces_list.append(m_sces)
        m_sces_list.append(m_sces_list)

    # _apply_kmean_on_msces_with_trials(m_sces_list, n_trials, n_clusters, fct_to_keep_best_silhouettes)
    surrogate_silhouettes = np.hstack(pool.map(_apply_kmean_on_msces_with_trials, m_sces_list)),
    print(f"Before closure")
    # close down the pool and join
    pool.close()
    pool.join()
    pool.clear()

    if debug_mode:
        print(f'End of clustering on shuffled data with multi processing')
        # print(f"surrogate_silhouettes {surrogate_silhouettes}, perc_threshold {perc_threshold}")
    percentile_result = np.percentile(surrogate_silhouettes, perc_threshold)
    return percentile_result


def clusters_on_sce_from_covnorm(cells_in_sce, range_n_clusters, fct_to_keep_best_silhouettes=np.mean,
                                 n_surrogate=1000, neurons_labels=None,
                                 perc_threshold_for_kmean_surrogates=95, debug_mode=False):
    """

    :param cells_in_sce:
    :param range_n_clusters:
    :param fct_to_keep_best_silhouettes: function used to keep the best trial, will be applied on all the silhouette
    scores of each trials, the max will be kept
    :return:
    """

    m_sces = cells_in_sce
    # normalized covariance matrix
    m_sces = covnorm(m_sces)

    # key is the nth clusters as int, value is a list of list of SCE
    # (each list representing a cluster, so we have as many list as the number of cluster wanted)
    dict_best_clusters = dict()
    for i in range_n_clusters:
        dict_best_clusters[i] = []

    # nb of time to apply one given number of cluster
    n_trials = 100
    best_kmeans_by_cluster = dict()
    surrogate_percentiles_by_n_cluster = dict()
    # will keep the best silhouette score (depending on the fct_to_keep_best_silhouettes) for each number of cluster
    best_silhouette_score_by_n_cluster = np.zeros(np.max(range_n_clusters) + 1)
    for n_clusters in range_n_clusters:
        # if debug_mode:
        # print(f"Ongoing K-mean with {n_clusters} clusters")
        # surrogate_start_time = time.time()
        surrogate_percentile = surrogate_clustering(m_sces=m_sces, n_clusters=n_clusters,
                                                    n_surrogate=n_surrogate,
                                                    n_trials=100,
                                                    fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                                                    perc_threshold=perc_threshold_for_kmean_surrogates,
                                                    debug_mode=debug_mode)
        surrogate_stop_time = time.time()
        # print(" ")
        # print(f"## Time for surrogate "
        #       f"{np.round(surrogate_stop_time - surrogate_start_time, 3)} s")
        # print(" ")

        best_kmeans = None
        best_local_clusters_silhouette = None
        best_silhouettes = 0
        # print(f"Start clustering on data with {n_clusters} clusters: {n_trials} trials")
        for trial in np.arange(n_trials):
            kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
            cluster_labels = kmeans.labels_
            sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
            local_clusters_silhouette = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values

            # compute a score based on the silhouette of each cluster for this trial and compare it with the best score
            # so far, keeping it if it's better
            computed_score = fct_to_keep_best_silhouettes(local_clusters_silhouette)
            # silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')

            if computed_score > best_silhouettes:
                best_silhouettes = computed_score
                best_silhouette_score_by_n_cluster[n_clusters] = computed_score
                best_local_clusters_silhouette = local_clusters_silhouette
                best_kmeans = kmeans

        best_kmeans_by_cluster[n_clusters] = best_kmeans
        surrogate_percentiles_by_n_cluster[n_clusters] = surrogate_percentile

        best_local_clusters_silhouette = [f"{v:.4f}" for v in best_local_clusters_silhouette]
        best_local_clusters_silhouette = ",".join(best_local_clusters_silhouette)
        print(f"For {n_clusters} clusters, "
              f"best avg silh-value: {best_silhouette_score_by_n_cluster[n_clusters]:.4f}, "
              f"with: {best_local_clusters_silhouette}, "
              f"and {perc_threshold_for_kmean_surrogates}th percentile surrogate: {surrogate_percentile:.4f}")

    return best_kmeans_by_cluster, m_sces, surrogate_percentiles_by_n_cluster, \
           best_silhouette_score_by_n_cluster, np.argmax(best_silhouette_score_by_n_cluster)


def plot_silhouettes(ax0, kmeans, m_sces, n_clusters, surrogate_silhouette_avg):
    cluster_labels = kmeans.labels_
    sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels)
    ax0.set_facecolor("black")
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # color = plt.nipy_spectral(float(i + 1) / (n_clusters + 1))
        color = BREWER_COLORS[i % len(BREWER_COLORS)]
        ax0.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=1)

        # putting a white edge around significan silhouette
        if surrogate_silhouette_avg is not None:
            if np.mean(ith_cluster_silhouette_values) > surrogate_silhouette_avg:
                ax0.plot(ith_cluster_silhouette_values, np.arange(y_lower, y_upper),
                         color="white", linewidth=1)
                # horizontal white line
                ax0.hlines(y_upper - 1, 0, ith_cluster_silhouette_values[-1],
                           color="white", linewidth=1)

        # Label the silhouette plots with their cluster numbers at the middle
        ax0.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color="white")

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax0.set_title("The silhouette plot for the various clusters.")
    ax0.set_xlabel("The silhouette coefficient values")
    ax0.set_ylabel("Cluster label")

    silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
    if surrogate_silhouette_avg is not None:
        ax0.axvline(x=surrogate_silhouette_avg, color="white", linestyle="--")
    # The vertical line for average silhouette score of all the values
    ax0.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax0.set_yticks([])  # Clear the yaxis labels / ticks
    ax0.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_covnorm_matrix(ax1, m_sces, n_clusters, cluster_labels):
    # display the normlized covariance matrix organized by cluster of SCE such as detected by initial kmeans
    # contains the neurons from the SCE, but ordered by cluster
    ordered_m_sces = np.zeros((np.shape(m_sces)[0], np.shape(m_sces)[1]))
    # to plot line that separate clusters
    cluster_coord_thresholds = []
    cluster_x_ticks_coord = []
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_m_sces[start:start + nb_k, :] = m_sces[e, :]
        ordered_m_sces[:, start:start + nb_k] = m_sces[:, e]
        start += nb_k
        if (k + 1) < n_clusters:
            if k == 0:
                cluster_x_ticks_coord.append(start / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_coord_thresholds[-1]) / 2)
            cluster_coord_thresholds.append(start)
        else:
            cluster_x_ticks_coord.append((start + cluster_coord_thresholds[-1]) / 2)

    co_var = np.corrcoef(ordered_m_sces)  # cov
    # sns.set()
    result = sns.heatmap(co_var, cmap="Blues", ax=ax1)  # , vmin=0, vmax=1) YlGnBu  cmap="jet" Blues
    # ax1.hlines(cluster_coord_thresholds, 0, np.shape(co_var)[0], color="black", linewidth=1,
    #            linestyles="dashed")
    for n_c, clusters_threshold in enumerate(cluster_coord_thresholds):
        # if (n_c+1) == len(cluster_coord_thresholds):
        #     break
        x_begin = 0
        if n_c > 0:
            x_begin = cluster_coord_thresholds[n_c - 1]
        x_end = np.shape(co_var)[0]
        if n_c < len(cluster_coord_thresholds) - 1:
            x_end = cluster_coord_thresholds[n_c + 1]
        ax1.hlines(clusters_threshold, x_begin, x_end, color="black", linewidth=2,
                   linestyles="dashed")
    for n_c, clusters_threshold in enumerate(cluster_coord_thresholds):
        # if (n_c+1) == len(cluster_coord_thresholds):
        #     break
        y_begin = 0
        if n_c > 0:
            y_begin = cluster_coord_thresholds[n_c - 1]
        y_end = np.shape(co_var)[0]
        if n_c < len(cluster_coord_thresholds) - 1:
            y_end = cluster_coord_thresholds[n_c + 1]
        ax1.vlines(clusters_threshold, y_begin, y_end, color="black", linewidth=2,
                   linestyles="dashed")
    # ax1.xaxis.get_majorticklabels().set_rotation(90)
    # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    # plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    ax1.set_xticks(cluster_x_ticks_coord)
    ax1.set_xticklabels(np.arange(n_clusters))
    ax1.set_yticks(cluster_x_ticks_coord)
    ax1.set_yticklabels(np.arange(n_clusters))
    ax1.set_title(f"{np.shape(m_sces)[0]} SCEs")
    # ax1.xaxis.set_tick_params(labelsize=5)
    # ax1.yaxis.set_tick_params(labelsize=5)
    ax1.invert_yaxis()


# TODO: do shuffling before the real cluster
# TODO: when showing co-var, show the 95th percentile of shuffling, to see which cluster is significant


def show_co_var_first_matrix(cells_in_peak, m_sces, n_clusters, kmeans, cluster_labels_for_neurons,
                             significant_sce_clusters,
                             data_str, path_results=None, show_fig=False, show_silhouettes=False,
                             surrogate_silhouette_avg=None, neurons_labels=None, axes_list=None, fig_to_use=None,
                             save_formats="pdf"):
    n_cells = len(cells_in_peak)

    if axes_list is None:
        if show_silhouettes:
            fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=False,
                                                gridspec_kw={'height_ratios': [1], 'width_ratios': [6, 6, 10]},
                                                figsize=(20, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False,
                                           gridspec_kw={'height_ratios': [1], 'width_ratios': [4, 10]},
                                           figsize=(20, 12))
        plt.tight_layout(pad=3, w_pad=7, h_pad=3)
        # ax1 = plt.subplot(121)
        plt.title(f"{data_str} {n_clusters} clusters")
    else:
        if show_silhouettes:
            ax0, ax1, ax2 = axes_list
        else:
            ax1, ax2 = axes_list
    # list of size nb_sce, each sce having a value from 0 to k clusters
    cluster_labels = kmeans.labels_

    # ################
    # silhouettes plot
    # ################
    if show_silhouettes:
        # Compute the silhouette scores for each sample
        plot_silhouettes(ax0=ax0, kmeans=kmeans, m_sces=m_sces, n_clusters=n_clusters,
                         surrogate_silhouette_avg=surrogate_silhouette_avg)

    # ############################
    # Normalized covariance plot
    # ############################
    plot_covnorm_matrix(ax1=ax1, m_sces=m_sces, n_clusters=n_clusters, cluster_labels=cluster_labels)

    # ############################
    # Heatmap cells vs SCE
    # ############################

    original_cells_in_peak = cells_in_peak

    cells_in_peak = np.copy(original_cells_in_peak)
    # ax2 = plt.subplot(1, 2, 2)

    # first order assemblies
    ordered_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]), dtype="int16")
    # then order neurons
    ordered_n_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]), dtype="int16")
    cluster_vertical_thresholds = []
    cluster_x_ticks_coord = []
    cluster_horizontal_thresholds = []
    # key is the cluster number and value is a tuple of int
    clusters_coords_dict = dict()
    cells_cluster_dict = dict()
    # set the number of neurons for whom there are no spikes or less than 2 for a given cluster
    nb_neurons_without_clusters = 0
    # if True, will put spike in each cluster in the color of the cluster, by putting the matrix value to the value of
    # the cluster, if False, will set in a darker color the cluster that belong to a cell
    color_each_clusters = True
    neurons_normal_order = np.arange(np.shape(cells_in_peak)[0])
    neurons_ax_labels = np.zeros(np.shape(cells_in_peak)[0], dtype="int16")
    # key is the cluster number, k, and value is an np.array of int reprenseting the indices of SCE part of this cluster
    sce_indices_for_each_clusters = dict()
    new_sce_labels = np.zeros(np.shape(cells_in_peak)[1], dtype="int16")
    nb_cells_by_cluster_of_cells = []
    nb_cells_by_cluster_of_cells_y_coord = []
    start = 0
    # first we put the sce that are not significant
    non_significant_sce_cluster = np.setdiff1d(np.arange(n_clusters), significant_sce_clusters)
    for sce_cluster_id in non_significant_sce_cluster:
        e = np.equal(cluster_labels, sce_cluster_id)
        nb_k = np.sum(e)
        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        sce_indices_for_each_clusters[sce_cluster_id] = np.arange(start, start + nb_k)
        old_pos = np.where(e)[0]
        for i, sce_index in enumerate(np.arange(start, start + nb_k)):
            new_sce_labels[sce_index] = old_pos[i]
        start += nb_k

    start_coord_significant_sce_clusters = start

    for index_k, k in enumerate(significant_sce_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        if color_each_clusters:
            for cell in np.arange(len(cells_in_peak)):
                spikes_index = np.where(cells_in_peak[cell, e] == 1)[0]
                to_put_to_true = np.where(e)[0][spikes_index]
                tmp_e = np.zeros(len(e), dtype="bool")
                tmp_e[to_put_to_true] = True
                # K +2 to avoid zero and one, and at the end we will substract 1
                cells_in_peak[cell, tmp_e] = k + 2

        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        sce_indices_for_each_clusters[k] = np.arange(start, start + nb_k)
        old_pos = np.where(e)[0]
        for i, sce_index in enumerate(np.arange(start, start + nb_k)):
            new_sce_labels[sce_index] = old_pos[i]
        start += nb_k

        if (index_k + 1) < len(significant_sce_clusters):
            if index_k == 0:
                cluster_x_ticks_coord.append((start + start_coord_significant_sce_clusters) / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_vertical_thresholds[-1]) / 2)
            cluster_vertical_thresholds.append(start)
        else:
            if len(cluster_vertical_thresholds) == 0:
                cluster_x_ticks_coord.append((start + start_coord_significant_sce_clusters) / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_vertical_thresholds[-1]) / 2)

    start = 0

    for k in np.arange(-1, np.max(cluster_labels_for_neurons) + 1):
        e = np.equal(cluster_labels_for_neurons, k)
        nb_cells = np.sum(e)
        if nb_cells == 0:
            # print(f'n_clusters {n_clusters}, k {k} nb_cells == 0')
            continue
        # print(f'nb_k {nb_k}, k: {k}')
        if k == -1:
            nb_neurons_without_clusters = nb_cells
        else:
            if not color_each_clusters:
                sce_indices = np.array(sce_indices_for_each_clusters[k])
                # print(f"sce_indices {sce_indices}, np.shape(ordered_cells_in_peak) {np.shape(ordered_cells_in_peak)}, "
                #       f"e {e} ")
                # we put to a value > 1 the sce where the neuron has a spike in their assigned cluster
                # to_modify = ordered_cells_in_peak[e, :][:, mask]
                for index in sce_indices:
                    tmp_e = np.copy(e)
                    # keeping for each sce, the cells that belong to cluster k
                    tmp_array = ordered_cells_in_peak[tmp_e, index]
                    # finding which cells don't have spikes
                    pos = np.where(tmp_array == 0)[0]
                    to_put_to_false = np.where(tmp_e)[0][pos]
                    tmp_e[to_put_to_false] = False
                    # putting to 2 all cells for whom there is a spike
                    ordered_cells_in_peak[tmp_e, index] = 2
                # to_modify[np.where(to_modify)[0]] = 2
        ordered_n_cells_in_peak[start:start + nb_cells, :] = ordered_cells_in_peak[e, :]
        neurons_ax_labels[start:start + nb_cells] = neurons_normal_order[e]
        nb_cells_by_cluster_of_cells.append(nb_cells)
        nb_cells_by_cluster_of_cells_y_coord.append(start + (nb_cells / 2))
        for cell in np.arange(start, start + nb_cells):
            cells_cluster_dict[cell] = k
        clusters_coords_dict[k] = (start, start + nb_cells)
        start += nb_cells
        if (k + 1) < (np.max(cluster_labels_for_neurons) + 1):
            cluster_horizontal_thresholds.append(start)

    if color_each_clusters:
        # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
        ordered_n_cells_in_peak = ordered_n_cells_in_peak - 2
        # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
        list_color = ['black', 'white']
        bounds = [-2.5, -1.5, -0.5]
        for i in np.arange(n_clusters):
            # color = plt.nipy_spectral(float(i + 1) / (n_clusters + 1))
            color = BREWER_COLORS[i % len(BREWER_COLORS)]
            list_color.append(color)
            bounds.append(i + 0.5)
        cmap = mpl.colors.ListedColormap(list_color)
    else:
        # light_blue_color = [0, 0.871, 0.219]
        cmap = mpl.colors.ListedColormap(['black', 'cornflowerblue', 'blue'])
        # cmap.set_over('red')
        # cmap.set_under('blue')
        bounds = [-0.5, 0.5, 1.5, 2.5]

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # rasterized=True used to remove the grid
    sns.heatmap(ordered_n_cells_in_peak, cbar=False, ax=ax2, cmap=cmap, norm=norm, rasterized=True)
    # print(f"len(neurons_ax_labels) {len(neurons_ax_labels)}")
    # TODO: set the position of labels, right now only one on two are displayed, fontsize should be decreased
    if neurons_labels is not None:
        ordered_neurons_labels = []
        for index in neurons_ax_labels:
            ordered_neurons_labels.append(neurons_labels[index])
        ax2.set_yticks(np.arange(len(ordered_neurons_labels)) + 0.5)
        ax2.set_yticklabels(ordered_neurons_labels)

    else:
        ax2.set_yticks(np.arange(len(neurons_ax_labels)))
        ax2.set_yticklabels(neurons_ax_labels.astype(int))

    if len(neurons_ax_labels) > 100:
        ax2.yaxis.set_tick_params(labelsize=4)
    elif len(neurons_ax_labels) > 200:
        ax2.yaxis.set_tick_params(labelsize=3)
    elif len(neurons_ax_labels) > 400:
        ax2.yaxis.set_tick_params(labelsize=2)
    else:
        ax2.yaxis.set_tick_params(labelsize=8)

    # creating axis at the top
    ax_top = ax2.twiny()
    ax_right = ax2.twinx()
    ax2.set_frame_on(False)
    ax_top.set_frame_on(False)
    ax_top.set_xlim((0, np.shape(cells_in_peak)[1]))
    ax_top.set_xticks(cluster_x_ticks_coord)
    # clusters labels
    ax_top.set_xticklabels(significant_sce_clusters)

    # print(f"nb_cells_by_cluster_of_cells_y_coord {nb_cells_by_cluster_of_cells_y_coord} "
    #       f"nb_cells_by_cluster_of_cells {nb_cells_by_cluster_of_cells}")
    ax_right.set_frame_on(False)
    ax_right.set_ylim((0, len(neurons_ax_labels)))
    ax_right.set_yticks(nb_cells_by_cluster_of_cells_y_coord)
    # clusters labels
    ax_right.set_yticklabels(nb_cells_by_cluster_of_cells)

    ax2.set_xticks(np.arange(np.shape(cells_in_peak)[1]) + 0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    # sce labels
    ax2.set_xticklabels(new_sce_labels)
    if len(new_sce_labels) > 100:
        ax2.xaxis.set_tick_params(labelsize=4)
    elif len(new_sce_labels) > 200:
        ax2.xaxis.set_tick_params(labelsize=3)
    elif len(new_sce_labels) > 400:
        ax2.xaxis.set_tick_params(labelsize=2)
    else:
        ax2.xaxis.set_tick_params(labelsize=6)
    ax2.hlines(cluster_horizontal_thresholds, start_coord_significant_sce_clusters, np.shape(cells_in_peak)[1],
               color="red", linewidth=1,
               linestyles="dashed")
    ax2.vlines([start_coord_significant_sce_clusters] + cluster_vertical_thresholds, 0,
               np.shape(cells_in_peak)[0], color="red", linewidth=1,
               linestyles="dashed")
    # print(f"n_clusters {n_clusters}, cluster_vertical_thresholds {cluster_vertical_thresholds}")
    for index, cluster in enumerate(significant_sce_clusters):
        if cluster not in clusters_coords_dict:
            # print(f"cluster {cluster} with no cells")
            # means no cell has this cluster as main cluster
            continue
        y_bottom, y_top = clusters_coords_dict[cluster]
        x_left = start_coord_significant_sce_clusters if (index == 0) else cluster_vertical_thresholds[index - 1]
        x_right = np.shape(cells_in_peak)[1] if (index == (len(significant_sce_clusters) - 1)) \
            else cluster_vertical_thresholds[index]
        linewidth = 3
        color_border = "white"
        ax2.vlines(x_left, y_bottom, y_top, color=color_border, linewidth=linewidth)
        ax2.vlines(x_right, y_bottom, y_top, color=color_border, linewidth=linewidth)
        ax2.hlines(y_bottom, x_left, x_right, color=color_border, linewidth=linewidth)
        ax2.hlines(y_top, x_left, x_right, color=color_border, linewidth=linewidth)

    # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)

    for cell in np.arange(n_cells):
        cluster = cells_cluster_dict[cell]
        if cluster >= 0:
            # color = plt.nipy_spectral(float(cluster + 1) / (n_clusters + 1))
            color = BREWER_COLORS[cluster % len(BREWER_COLORS)]
            ax2.get_yticklabels()[cell].set_color(color)

    ax2.invert_yaxis()
    # if nb_neurons_without_clusters > 0:
    #     for i in np.arange(nb_neurons_without_clusters):
    #         ax2.get_yticklabels()[i].set_color("red")

    if (path_results is not None) and ((axes_list is None) or (fig_to_use is not None)):
        if fig_to_use is not None:
            fig = fig_to_use
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{data_str}_{n_clusters}_sce_clusters.{save_format}',
                        format=f"{save_format}")
    if show_fig:
        plt.show()
        plt.close()


def give_significant_sce_clusters(kmeans_dict, range_n_clusters, m_sces, surrogate_percentile):
    significant_sce_clusters = dict()
    for n_cluster in range_n_clusters:
        # print(f"Results from K-means with {n_cluster} clusters: Look for significant clusters")
        if kmeans_dict[n_cluster] is None:
            continue
        cluster_labels = kmeans_dict[n_cluster].labels_
        significant_sce_clusters[n_cluster] = []

        # keeping only the significant clusters
        sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')

        for cluster_id in range(n_cluster):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == cluster_id]
            avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
            # print(f"cluster_id {cluster_id}: {avg_ith_cluster_silhouette_values} vs surrogate {surrogate_percentile[n_cluster]}")
            if avg_ith_cluster_silhouette_values > surrogate_percentile[n_cluster]:
                # then we keep it
                significant_sce_clusters[n_cluster].append(cluster_id)
        significant_sce_clusters[n_cluster] = np.array(significant_sce_clusters[n_cluster])

    return significant_sce_clusters


def find_cluster_labels_for_neurons(cells_in_peak, cluster_labels, m_sces, significant_clusters):
    """
    Building cell_assemblies
    :param cells_in_peak:
    :param cluster_labels:
    :param m_sces:
    :param significant_clusters:
    :return: cluster_labels_for_neurons np.array of len n_cells, each value correpond to the cell assembly
    the cell belongs, if the value equal -1, then the cell doesn't belong to any cell assemblie
    """
    cluster_labels_for_neurons = np.zeros(np.shape(cells_in_peak)[0], dtype="int8")
    # sorting neurons spikes, keeping them only in one cluster, the one with the most percentage spikes from this neuron
    # only from sce clusters significant
    # if spikes < 2 in any clusters, then removing spikes
    # going neuron by neuron,

    for n, events in enumerate(cells_in_peak):
        if len(significant_clusters) == 0:
            cluster_labels_for_neurons[n] = -1
            continue
        pos_events = np.where(events)[0]
        # will contains the number of spike in each sce for the neuron n
        max_clusters = np.zeros(len(significant_clusters), dtype="float")
        for p in pos_events:
            # p correspond to the index of one SCE
            sce_cluster_of_p = cluster_labels[p]
            if sce_cluster_of_p in significant_clusters:
                index = np.where(significant_clusters == sce_cluster_of_p)[0][0]
                max_clusters[index] += 1
        if np.max(max_clusters) < 2:
            # it should spike at least 2 times in a SCE cluster to be part of it
            cluster_labels_for_neurons[n] = -1
        else:
            # putting max_clusters into percentages, it means what percentages of sce the cell is spiking on
            max_clusters_perc = np.copy(max_clusters)
            for i in np.arange(len(max_clusters)):
                # i is the index of a sce cluster in significant_clusters
                nb_sce_in_cluster = len(np.where(cluster_labels == significant_clusters[i])[0])
                max_clusters_perc[i] = (max_clusters[i] / nb_sce_in_cluster) * 100
            # selecting the cluster with the most spikes from neuron n
            max_value = np.max(max_clusters_perc)
            pos_max_value = np.where(max_clusters_perc == max_value)[0]
            if len(pos_max_value) > 1:
                # if more than one sce with the max, we keep the one with the more spikes in it
                max_spikes = 0
                best_pos = 0
                for pos in pos_max_value:
                    if max_spikes < max_clusters[pos]:
                        max_spikes = max_clusters[pos]
                        best_pos = pos
                pos_max_value = best_pos
            else:
                pos_max_value = pos_max_value[0]
            # max_cluster = np.argmax(max_clusters_perc)
            cluster_labels_for_neurons[n] = significant_clusters[pos_max_value]
            # clearing spikes from other cluster
            # if removing_multiple_spikes_among_cluster:
            #     cells_in_peak[n, np.not_equal(cluster_labels, max_cluster)] = 0
    return cluster_labels_for_neurons


def save_stat_SCE_and_cluster_k_mean_version(spike_nums_to_use, activity_threshold, k_means,
                                             SCE_times, n_cluster, path_results, sliding_window_duration,
                                             cluster_labels_for_neurons, perc_threshold,
                                             n_surrogate_k_mean, data_descr,
                                             n_surrogate_activity_threshold):
    round_factor = 2
    file_name = f'{path_results}/{data_descr}_{n_cluster}_clusters_stat_k_mean_.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Stat k_mean version for {n_cluster} clusters" + '\n')
        file.write("" + '\n')
        file.write(f"cells {len(spike_nums_to_use)}, events {len(SCE_times)}" + '\n')
        file.write(f"Event participation threshold {activity_threshold}, {perc_threshold} percentile, "
                   f"{n_surrogate_activity_threshold} surrogates" + '\n')
        file.write(f"Sliding window duration {sliding_window_duration}" + '\n')
        file.write(f"{n_surrogate_k_mean} surrogates for kmean" + '\n')
        file.write("" + '\n')
        file.write("" + '\n')
        cluster_labels = k_means.labels_

        for k in np.arange(n_cluster):

            e = np.equal(cluster_labels, k)

            nb_sce_in_cluster = np.sum(e)
            sce_ids = np.where(e)[0]

            e_cells = np.equal(cluster_labels_for_neurons, k)
            n_cells_in_cluster = np.sum(e_cells)

            file.write("#" * 10 + f"   cluster {k} / {nb_sce_in_cluster} events / {n_cells_in_cluster} cells" +
                       "#" * 10 + '\n')
            file.write('\n')

            duration_values = np.zeros(nb_sce_in_cluster, dtype="uint16")
            max_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")
            mean_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")
            overall_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")

            for n, sce_id in enumerate(sce_ids):
                duration_values[n], max_activity_values[n], \
                mean_activity_values[n], overall_activity_values[n] = \
                    give_stat_one_sce(sce_id=sce_id,
                                      spike_nums_to_use=spike_nums_to_use,
                                      SCE_times=SCE_times, sliding_window_duration=sliding_window_duration)
            file.write(f"Duration: mean {np.round(np.mean(duration_values), round_factor)}, "
                       f"std {np.round(np.std(duration_values), round_factor)}, "
                       f"median {np.round(np.median(duration_values), round_factor)}\n")
            file.write(f"Overall participation: mean {np.round(np.mean(overall_activity_values), round_factor)}, "
                       f"std {np.round(np.std(overall_activity_values), round_factor)}, "
                       f"median {np.round(np.median(overall_activity_values), round_factor)}\n")
            file.write(f"Max participation: mean {np.round(np.mean(max_activity_values), round_factor)}, "
                       f"std {np.round(np.std(max_activity_values), round_factor)}, "
                       f"median {np.round(np.median(max_activity_values), round_factor)}\n")
            file.write(f"Mean participation: mean {np.round(np.mean(mean_activity_values), round_factor)}, "
                       f"std {np.round(np.std(mean_activity_values), round_factor)}, "
                       f"median {np.round(np.median(mean_activity_values), round_factor)}\n")
            file.write('\n')

        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')
        # for each SCE
        for sce_id in np.arange(len(SCE_times)):
            result = give_stat_one_sce(sce_id=sce_id,
                                       spike_nums_to_use=spike_nums_to_use,
                                       SCE_times=SCE_times,
                                       sliding_window_duration=sliding_window_duration)
            duration_in_frames, max_activity, mean_activity, overall_activity = result
            file.write(f"SCE {sce_id}" + '\n')
            file.write(f"Duration_in_frames {duration_in_frames}" + '\n')
            file.write(f"Overall participation {np.round(overall_activity, round_factor)}" + '\n')
            file.write(f"Max participation {np.round(max_activity, round_factor)}" + '\n')
            file.write(f"Mean participation {np.round(mean_activity, round_factor)}" + '\n')

            file.write('\n')
            file.write('\n')


def give_stat_one_sce(sce_id, spike_nums_to_use, SCE_times, sliding_window_duration):
    """

    :param sce_id:
    :param spike_nums_to_use:
    :param SCE_times:
    :param sliding_window_duration:
    :return: duration_in_frames: duration of the sce in frames
     max_activity: the max number of cells particpating during a window_duration to the sce
     mean_activity: the mean of cells particpating during the sum of window_duration
     overall_activity: the number of different cells participating to the SCE all along
     if duration == sliding_window duration, max_activity, mean_activity and overall_activity will be equal
    """
    time_tuple = SCE_times[sce_id]
    duration_in_frames = (time_tuple[1] - time_tuple[0]) + 1
    n_slidings = (duration_in_frames - sliding_window_duration) + 1

    sum_activity_for_each_frame = np.zeros(n_slidings)
    for n in np.arange(n_slidings):
        # see to use window_duration to find the amount of participation
        time_beg = time_tuple[0] + n
        sum_activity_for_each_frame[n] = len(np.where(np.sum(spike_nums_to_use[:,
                                                             time_beg:(time_beg + sliding_window_duration)],
                                                             axis=1))[0])
    max_activity = np.max(sum_activity_for_each_frame)
    mean_activity = np.mean(sum_activity_for_each_frame)
    overall_activity = len(np.where(np.sum(spike_nums_to_use[:,
                                           time_tuple[0]:(time_tuple[1] + 1)], axis=1))[0])

    return duration_in_frames, max_activity, mean_activity, overall_activity


def statistical_cell_assemblies_def(cell_assemblies_struct,
                                    n_surrogate=1000, debug_mode=False):
    """
    :param sce_clusters_id:
    :param sce_clusters_labels:
    :param cellsinpeak:
    :return: will modify the cell_assemblies_struct instance and update the cellsinpeak_ordered and other useful
    variables used for display purpose
    """
    """
    Each cluster of SCE was then associated to a cell assembly which comprised those cells that significantly 
    participated to the SCE events within that particular cluster. 
    Cell participation to a given cluster was considered statistically significant if the fraction of SCE in 
    that cluster that activated the cell exceeded the 95th percentile of reshuffled data. 
    If a cell was significantly active in more than one SCE cluster, 
    it was associated to the one in which it participated the most (percentage wise).
    """
    # Count number of participation to each cluster
    # CellP means cell positives, and R means ratio (CellR gives for each cell the
    # percentage of sce during which it's active inside a cluster of SCE.
    cas = cell_assemblies_struct
    # if debug_mode:
    #     for n_id, neuron_label in enumerate(cas.neurons_labels):
    #         print(f"{n_id} -> {neuron_label}")
    sce_clusters_id = cas.sce_clusters_id
    sce_clusters_labels = cas.sce_clusters_labels
    cellsinpeak = cas.cellsinpeak
    n_cells = cas.n_cells
    n_sce_clusters = cas.n_clusters
    n_sces = cas.n_sces
    cells_p = np.zeros((n_cells, n_sce_clusters))
    cells_r = np.zeros((n_cells, n_sce_clusters))
    for i, cluster_id in enumerate(sce_clusters_id):
        cells_p[:, i] = np.sum(cellsinpeak[:, np.equal(sce_clusters_labels, cluster_id)], axis=1)
        cells_r[:, i] = cells_p[:, i] / np.sum(np.equal(sce_clusters_labels, cluster_id))

    # if debug_mode:
    #     print(f"cells_p {cells_p}")
    #     for cell_id, cell_p in enumerate(cells_p):
    #         print(f"{cell_id}: cas.neurons_labels {cas.neurons_labels[cell_id]} {cells_p}")
    #     print(f"cells_r {cells_r}")

    # Test for statistical significance
    cells_cl = np.zeros((n_sce_clusters, n_cells), dtype="uint8")  # Binary matrix of cell associated to clusters

    for cell in np.arange(n_cells):
        # print(f"cell {cell} {cas.neurons_labels[cell]}:")
        # Random distribution among Clusters
        r_clr = np.zeros((n_sce_clusters, n_surrogate))
        # number of random values to apply, look at how many times the cell spikes in a SCE
        n_rnd = int(np.sum(cellsinpeak[cell, :]))
        # print(f"n_rnd {n_rnd}")

        for surrogate in np.arange(n_surrogate):
            rand_perm = np.random.permutation(n_sces)
            rand_perm = rand_perm[:n_rnd]
            racer = np.zeros(n_sces, dtype="uint8")
            racer[rand_perm] = 1
            for i, cluster_id in enumerate(sce_clusters_id):
                r_clr[i, surrogate] = np.sum(racer[np.equal(sce_clusters_labels, cluster_id)])
        # sorting r_clr, for each cluster, will sort by ascending number of spikes in a surrogate SCE
        r_clr = np.sort(r_clr, axis=1)
        # / n_sce_clusters is to account for the inter- dependence of the comparisons (Bonferroni correction)
        th_max = r_clr[:, int(n_surrogate * (1 - (0.05 / n_sce_clusters)))]
        # if debug_mode:
        # print(f"int(n_surrogate * (1 - 0.05 / n_sce_clusters)) {int(n_surrogate * (1 - 0.05 / n_sce_clusters))}")
        # print(f"th_max {th_max}")
        # print(f"cells_p[cell, :] {cells_p[cell, :]}")
        for i, cluster_id in enumerate(sce_clusters_id):
            cells_cl[i, cell] = int(cells_p[cell, i] > th_max[i])

    cells_with_no_sce_cluster = np.where(np.sum(cells_cl, axis=0) == 0)[0]  # Cells not in any cluster
    cells_with_one_sce_cluster = np.where(np.sum(cells_cl, axis=0) == 1)[0]
    cells_with_several_sce_cluster = np.where(np.sum(cells_cl, axis=0) >= 2)[0]

    # print(f"cells_cl {cells_cl}")
    if debug_mode:
        print(f"cells_with_no_sce_cluster {cells_with_no_sce_cluster}")
        print(f"cells_with_one_sce_cluster {cells_with_one_sce_cluster}")
        print(f"cells_with_several_sce_cluster {cells_with_several_sce_cluster}")

    # Keep cluster where they participate the most
    for cell_id in cells_with_several_sce_cluster:
        # If a cell was significantly active in more than one SCE cluster,
        # it was associated to the one in which it participated the most (percentage wise)
        index_cl = np.argmax(cells_r[cell_id, :])
        cells_cl[:, cell_id] = 0
        cells_cl[index_cl, cell_id] = 1

    if debug_mode:
        print(f"cells_cl {cells_cl}")

    """
    The overlap between assemblies was quantified by calculating the silhouette value of each cell 
    (with the normalized hamming distance between each cell pair as a dissimilarity metric). 
    A cell was significantly involved in a single assembly if its silhouette value was higher 
    than expected by chance (95th percentile after reshuffling). SCE were finally sorted with respect 
    to their projection onto cell assemblies. A SCE was activating a given neuronal assembly 
    if the number of cells recruited in that assembly was higher than expected by chance (95th percentile after reshuffling)
    """

    # list of list of int representing cell indices that
    # belong to this cluster
    # cell_assemblies_clusters represents the cell assemblies clusters
    # cell_assemblies_clusters is c_0 in Arnaud's code
    cell_assemblies_clusters = list()
    # dict used to map the cell assembly cluster index to the sce cluster index it was formed by
    map_index_cluster_to_original_cluster_id = dict()
    for i, cluster_id in enumerate(sce_clusters_id):
        min_cells_nb = 2
        cells_in_cluster = np.where(cells_cl[i, :])[0]
        # keep cells index for clusters with at least min_cells_nb
        if len(cells_in_cluster) >= min_cells_nb:
            map_index_cluster_to_original_cluster_id[len(cell_assemblies_clusters)] = cluster_id
            cell_assemblies_clusters.append(cells_in_cluster)

    # print(f"c_0 {cell_assemblies_clusters}")

    # Participation rate to its own cluster
    # not used
    cells_r_cl = np.max(cells_r[np.concatenate((cells_with_one_sce_cluster, cells_with_several_sce_cluster)), :],
                        axis=1)

    # Assign sce to groups of cells

    # n_assemblies_cl (NCl) is the number of significant clusters with more than x cells
    # C0 is a dict, key represent cluster number, and value is an array
    # of cells index belonging to the cluster
    n_assemblies_cl = len(cell_assemblies_clusters)
    if debug_mode:
        print(f"n_assemblies_cl {n_assemblies_cl}")
    # Cell count in each cluster
    r_cl = np.zeros((n_assemblies_cl, n_sces))
    p_cl = np.zeros((n_assemblies_cl, n_sces),
                    dtype="uint8")  # binary matrix of sce associated to cell assembly clusters

    for i, cells in enumerate(cell_assemblies_clusters):
        r_cl[i, :] = np.sum(cellsinpeak[cells, :], axis=0)

    # print(f"r_cl {r_cl}")

    r_cln = np.zeros((n_assemblies_cl, n_sces))

    for sce_id in np.arange(n_sces):
        # Random distribution among Clusters
        r_clr = np.zeros((n_assemblies_cl, n_surrogate))
        n_rnd = np.sum(cellsinpeak[:, sce_id])

        for surrogate_id in np.arange(n_surrogate):
            rand_perm = np.random.permutation(n_cells)
            rand_perm = rand_perm[:n_rnd]
            racer = np.zeros(n_cells, dtype="uint8")
            racer[rand_perm] = 1
            for cell_assembly_id, cells in enumerate(cell_assemblies_clusters):
                r_clr[cell_assembly_id, surrogate_id] = np.sum(racer[cells])

        # sorting r_clr, for each cluster, will sort by ascending number of spikes in a surrogate SCE
        r_clr = np.sort(r_clr, axis=1)
        # / n_sce_clusters is for the bonferroni correction
        th_max = r_clr[:, int(n_surrogate * (1 - (0.05 / n_sce_clusters)))]
        # print(f"int(n_surrogate * (1 - 0.05 / n_sce_clusters)) {int(n_surrogate * (1 - 0.05 / n_sce_clusters))}")
        # print(f"th_max {th_max}")
        # print(f"cells_p[j, :] {cells_p[j, :]}")
        for cell_assembly_id, cells in enumerate(cell_assemblies_clusters):
            p_cl[cell_assembly_id, sce_id] = int(r_cl[cell_assembly_id, sce_id] > th_max[cell_assembly_id])
        # Normalize (probability)
        r_cln[:, sce_id] = r_cl[:, sce_id] / np.sum(cellsinpeak[:, sce_id])
    if debug_mode:
        print(f"np.sum(p_cl, axis=1) {np.sum(p_cl, axis=1)}")

    ############################################
    # #### variables for later display purposes
    ############################################
    # give the original cell index
    cells_indices = np.zeros(n_cells, dtype="uint16")
    # give the original sce index
    sce_indices = np.zeros(n_sces, dtype="uint16")
    # give the number of sce in the no-assembly-sce, single-assembly and multiple-assembly groups respectively
    n_sce_in_assembly = np.zeros(3, dtype="uint16")
    # contains the nb of cells in each cell assemblie cluster
    n_cells_in_cell_assemblies_clusters = []
    # contains the nb of sces in sce single cell assembly cluster
    n_cells_in_single_cell_assembly_sce_cl = []
    # contains the nb of sces in sce multiple cell assembly cluster
    n_cells_in_multiple_cell_assembly_sce_cl = []
    # key will be the cell assembly index (such as displayed) and the value is a list of list of 2 elements (first and
    # last sce index(non included) of each sce cluster part of this assembly)
    sces_in_cell_assemblies_clusters = dict()
    # key is a int representing a sce, and value a list representing the id of cell assemblies cluster
    cell_assemblies_cluster_of_multiple_ca_sce = dict()

    cellsinpeak_just_cells_ordered = np.zeros((n_cells, n_sces), dtype="int16")

    # #### ordering cells first   #####
    all_cells = np.arange(n_cells)
    cells_ordered = np.zeros(0, dtype="int16")
    start = 0
    for cells in cell_assemblies_clusters:
        cellsinpeak_just_cells_ordered[start:(start + len(cells)), :] = cellsinpeak[cells, :]
        cells_indices[start:(start + len(cells))] = cells
        n_cells_in_cell_assemblies_clusters.append(len(cells))
        start += len(cells)
        cells_ordered = np.concatenate((cells_ordered, cells))
    cells_left = np.setdiff1d(all_cells, cells_ordered)
    cellsinpeak_just_cells_ordered[start:, :] = cellsinpeak[cells_left, :]
    cells_indices[start:] = cells_left

    cellsinpeak_ordered = np.zeros((n_cells, n_sces), dtype="int16")

    # now ordering sce
    # p_cl = np.zeros((n_cl, n_sces))
    # first we look at sces associated to no cell assembly
    start = 0
    sce_still_to_organized = np.arange(n_sces)
    no_assembly_sce = np.where(np.sum(p_cl, axis=0) == 0)[0]
    if debug_mode:
        print(f"len(no_assembly_sce) {len(no_assembly_sce)}, "
              f"no_assembly_sce {no_assembly_sce}")
    # toto error
    cellsinpeak_ordered[:, start:(start + len(no_assembly_sce))] = cellsinpeak_just_cells_ordered[:, no_assembly_sce]
    sce_indices[start:(start + len(no_assembly_sce))] = no_assembly_sce
    start += len(no_assembly_sce)
    n_sce_in_assembly[0] = len(no_assembly_sce)
    sce_still_to_organized = np.setdiff1d(sce_still_to_organized, no_assembly_sce)

    # then those sces associated to one cell assembly
    # and we organize sces among them as sce clusters in which the cell assemblie belongs
    single_assembly_sce = np.where(np.sum(p_cl, axis=0) == 1)[0]

    if debug_mode:
        print(f"len(single_assembly_sce) {len(single_assembly_sce)}, "
              f"single_assembly_sce {single_assembly_sce}")
    if len(single_assembly_sce) > 0:
        # putting them in the same order as cell assemblies
        sce_already_organized = np.zeros(0, dtype="uint16")
        for cell_assembly_cluster_id in np.arange(len(cell_assemblies_clusters)):
            # sce index that are part of this cluster
            sces_in_cluster = np.where(p_cl[cell_assembly_cluster_id, :])[0]
            # now we check which ones are indeed in the single_assembly_sce group
            sces_in_cluster = (np.intersect1d(sces_in_cluster, single_assembly_sce)).astype(int)
            # print(f"np.intersect1d(sces_in_cluster, sce_already_organized) "
            #       f"{np.intersect1d(sces_in_cluster, sce_already_organized)}")
            if len(sces_in_cluster) > 0:
                cellsinpeak_ordered[:, start:(start + len(sces_in_cluster))] = \
                    cellsinpeak_just_cells_ordered[:, sces_in_cluster]
                sce_indices[start:(start + len(sces_in_cluster))] = sces_in_cluster
                if cell_assembly_cluster_id not in sces_in_cell_assemblies_clusters:
                    sces_in_cell_assemblies_clusters[cell_assembly_cluster_id] = []
                sces_in_cell_assemblies_clusters[cell_assembly_cluster_id]. \
                    append((start, (start + len(sces_in_cluster))))

                start += len(sces_in_cluster)
                n_sce_in_assembly[1] += len(sces_in_cluster)
                n_cells_in_single_cell_assembly_sce_cl.append(len(sces_in_cluster))
                sce_already_organized = np.concatenate((sce_already_organized, sces_in_cluster))
        # print(f"np.setdiff1d(single_assembly_sce, sce_already_organized : "
        #       f"{np.setdiff1d(single_assembly_sce, sce_already_organized)}")
        sce_still_to_organized = np.setdiff1d(sce_still_to_organized, sce_already_organized)

    # then those sces associated to multiple cell assembly
    # and we organize sces among them as sce clusters in which the cell assemblie belongs
    multiple_assembly_sce = np.where(np.sum(p_cl, axis=0) > 1)[0]
    if debug_mode:
        print(f"len(multiple_assembly_sce) {len(multiple_assembly_sce)}, "
              f"multiple_assembly_sce {multiple_assembly_sce}")
    if len(multiple_assembly_sce) > 0:
        for sce_id in multiple_assembly_sce:
            cell_assemblies_cluster_of_multiple_ca_sce[sce_id] = np.where(p_cl[:, sce_id])[0]
        sce_already_organized = np.zeros(0, dtype="uint16")
        for cluster_id in np.arange(len(cell_assemblies_clusters)):
            # sce index that are part of this cluster
            sces_in_cluster = np.where(p_cl[cluster_id, :])[0]
            # now we check which ones are indeed in the multiple_assembly_sce group
            sces_in_cluster = (np.intersect1d(sces_in_cluster, multiple_assembly_sce)).astype(int)
            # because they are in multiple cell assemblies, we remove them if they have already been added to a cell
            # assembly cluster for ordering
            sces_in_cluster = (np.setdiff1d(sces_in_cluster, sce_already_organized)).astype(int)
            # print(f"np.intersect1d(sces_in_cluster, sce_already_organized) "
            #       f"{np.intersect1d(sces_in_cluster, sce_already_organized)}")
            if len(sces_in_cluster) > 0:
                cellsinpeak_ordered[:, start:(start + len(sces_in_cluster))] = \
                    cellsinpeak_just_cells_ordered[:, sces_in_cluster]
                sce_indices[start:(start + len(sces_in_cluster))] = sces_in_cluster
                # if cluster_id not in sces_in_cell_assemblies_clusters:
                #     sces_in_cell_assemblies_clusters[cluster_id] = []
                # sces_in_cell_assemblies_clusters[cluster_id].append((start, (start + len(sces_in_cluster))))
                start += len(sces_in_cluster)
                n_sce_in_assembly[2] += len(sces_in_cluster)
                n_cells_in_multiple_cell_assembly_sce_cl.append(len(sces_in_cluster))
                sce_already_organized = np.concatenate((sce_already_organized, sces_in_cluster))
        # print(f"np.setdiff1d(multiple_assembly_sce, sce_already_organized : "
        #       f"{np.setdiff1d(multiple_assembly_sce, sce_already_organized)}")
        sce_still_to_organized = np.setdiff1d(sce_still_to_organized, sce_already_organized)

    if len(sce_still_to_organized) > 0:
        print(f"Something's wrong, sce_still_to_organized {sce_still_to_organized}")

    cas.cellsinpeak_ordered = cellsinpeak_ordered
    cas.n_cells_in_cell_assemblies_clusters = n_cells_in_cell_assemblies_clusters
    cas.n_cells_not_in_cell_assemblies = cas.n_cells - np.sum(n_cells_in_cell_assemblies_clusters)
    cas.cells_indices = cells_indices
    cas.sce_indices = sce_indices
    cas.sces_in_cell_assemblies_clusters = sces_in_cell_assemblies_clusters
    cas.n_sce_in_assembly = n_sce_in_assembly
    cas.n_cells_in_single_cell_assembly_sce_cl = n_cells_in_single_cell_assembly_sce_cl
    cas.n_cells_in_multiple_cell_assembly_sce_cl = n_cells_in_multiple_cell_assembly_sce_cl
    cas.cell_assemblies_cluster_of_multiple_ca_sce = cell_assemblies_cluster_of_multiple_ca_sce


def compute_kmean(neurons_labels, cellsinpeak, n_surrogates_k_mean, n_trials_k_mean,
                  min_n_assemblies, activity_threshold,
                  range_n_clusters, path_results,
                  perc_threshold_for_kmean_surrogates,
                  sliding_window_duration, SCE_times, data_id,
                  fct_to_keep_best_silhouettes=np.mean,
                  keep_only_the_best=False,
                  debug_mode=False):
    """

    :param neurons_labels:
    :param cellsinpeak:
    :param n_surrogates_k_mean: int or list of 2 int, indicate how many surrogates should be used to determine if a
    k-mean results pass the threshold. If there are two values, then the first value indicate a low version,
    that is tried fast to save computational time, if it passed the threshold, then the second value is run for
    confirmation.
    :param n_trials_k_mean: int or list of 2 int, indicate how many trial of k-mean should be run to decide which
    cluster holds the best silhouette (the best orthogonality in a sort). It's different than finding a threshold.
    :param min_n_assemblies: if there are two number of surrogates for kmean, this is used to try the second if the
    minimum of assemblies has been found. The assemblies figure is also produce only if this given number is reached
    :param range_n_clusters:
    :param path_results:
    :param sliding_window_duration:
    :param SCE_times:
    :param data_id:
    :param fct_to_keep_best_silhouettes:
    :param keep_only_the_best:
    :param debug_mode:
    :return:
    """
    if isinstance(n_surrogates_k_mean, int):
        n_surrogate_k_mean = [n_surrogates_k_mean]

    if isinstance(n_trials_k_mean, int):
        n_trials_k_mean = [n_trials_k_mean]

    if len(n_surrogates_k_mean) != len(n_trials_k_mean):
        raise Exception("n_surrogates_k_mean & n_trials_k_mean should have the same length")

    significant_sce_clusters = dict()
    range_n_clusters_original = range_n_clusters

    for n_surrogate_k_mean, n_trial_k_mean in zip(n_surrogates_k_mean, n_trials_k_mean):
        print(" ")
        start_time = time.time()
        best_kmeans_by_cluster, m_cov_sces, surrogate_percentile, \
        best_silhouette_score_by_n_cluster, cluster_with_best_silhouette_score = \
            clusters_on_sce_from_covnorm(cells_in_sce=cellsinpeak,
                                         n_surrogate=n_surrogate_k_mean,
                                         fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                                         range_n_clusters=range_n_clusters,
                                         neurons_labels=neurons_labels,
                                         debug_mode=debug_mode,
                                         perc_threshold_for_kmean_surrogates=perc_threshold_for_kmean_surrogates)
        stop_time = time.time()
        print(f"## Time for clustering with {n_surrogate_k_mean} surrogates & {n_trial_k_mean} "
              f"trials on range {range_n_clusters}: "
              f"{(stop_time - start_time):.2f} s")
        print(" ")

        if keep_only_the_best:
            range_n_clusters = [cluster_with_best_silhouette_score]

        # give the indices of the sce cluster for whom the silhouette is > to the xth percentile from n_surrogate
        # surrogates
        # dict with key the number of clusters and value a list with the indices of the significant cluster
        significant_sce_clusters_tmp = give_significant_sce_clusters(kmeans_dict=best_kmeans_by_cluster,
                                                                     range_n_clusters=range_n_clusters,
                                                                     m_sces=m_cov_sces,
                                                                     surrogate_percentile=surrogate_percentile)

        significant_sce_clusters.update(significant_sce_clusters_tmp)
        # then we update the clusters range for next loop or break if None are left
        range_n_clusters = []
        for cluster_number, significant_clusters_list in significant_sce_clusters_tmp.items():
            if len(significant_clusters_list) >= min_n_assemblies:
                range_n_clusters.append(cluster_number)

        if len(range_n_clusters) == 0:
            break

    # if not keep_only_the_best:
    #     range_n_clusters = range_n_clusters_original
    #
    # # print(f"range_n_clusters {range_n_clusters}")
    # # print(f"significant_sce_clusters {significant_sce_clusters}")
    # # updating range_n_clusters
    # range_n_clusters_tmp = []
    # for n_cluster in range_n_clusters:
    #     if n_cluster in significant_sce_clusters:
    #         range_n_clusters_tmp.append(n_cluster)
    # range_n_clusters = np.array(range_n_clusters_tmp)

    cell_assemblies_struct_dict = dict()
    for n_cluster in range_n_clusters:
        # print(f"Results from K-means with {n_cluster} clusters: Test statistical significance of cell-assemblies")
        kmeans = best_kmeans_by_cluster[n_cluster]
        if n_cluster < len(best_silhouette_score_by_n_cluster):
            silhouette_score = best_silhouette_score_by_n_cluster[n_cluster]
        else:
            silhouette_score = 0
        cas = CellAssembliesStruct(data_id=data_id, sce_clusters_labels=kmeans.labels_, cellsinpeak=cellsinpeak,
                                   silhouette_score=silhouette_score, activity_threshold=activity_threshold,
                                   sce_clusters_id=significant_sce_clusters[n_cluster], n_clusters=n_cluster,
                                   cluster_with_best_silhouette_score=cluster_with_best_silhouette_score,
                                   path_results=path_results, neurons_labels=neurons_labels,
                                   sliding_window_duration=sliding_window_duration,
                                   n_surrogate_k_mean=n_surrogate_k_mean, SCE_times=SCE_times)
        start_time = time.time()
        statistical_cell_assemblies_def(cell_assemblies_struct=cas, debug_mode=debug_mode, n_surrogate=1000)
        print(f"For {n_cluster} clusters, {len(cas.n_cells_in_cell_assemblies_clusters)} assemblies found.")
        stop_time = time.time()
        cell_assemblies_struct_dict[n_cluster] = cas

    # will contains as value list of int (corresponding to sce cluster label, -1 if no sce cluster), the len of
    # the list is n_cells
    # So far, a cell belongs to a cluster if the cluster is significant, the cells spikes at least twice in it,
    # and among the significant cluster, the cell has the highest ratio of spikes in that one (for equal ratios, the
    # highest number of cells determine to which cluster it belongs
    cluster_labels_for_neurons = dict()
    for n_cluster in range_n_clusters:
        cluster_labels = best_kmeans_by_cluster[n_cluster].labels_
        cluster_labels_for_neurons[n_cluster] = \
            find_cluster_labels_for_neurons(cells_in_peak=cellsinpeak,
                                            cluster_labels=cluster_labels, m_sces=m_cov_sces,
                                            significant_clusters=significant_sce_clusters[n_cluster])

    return range_n_clusters, best_kmeans_by_cluster, m_cov_sces, cluster_labels_for_neurons, surrogate_percentile, \
           significant_sce_clusters, cluster_with_best_silhouette_score, cell_assemblies_struct_dict


def compute_and_plot_clusters_raster_kmean_version(labels, activity_threshold, range_n_clusters_k_mean,
                                                   n_surrogate_k_mean, k_mean_n_trials,
                                                   min_n_assemblies,
                                                   spike_nums_to_use, cellsinpeak, data_descr,
                                                   path_results,
                                                   sliding_window_duration, sce_times_numbers,
                                                   SCE_times, perc_threshold_for_sce,
                                                   n_surrogate_activity_threshold,
                                                   perc_threshold_for_kmean_surrogates,
                                                   params_to_save_dict,
                                                   keep_only_the_best,
                                                   with_shuffling=True,
                                                   sce_times_bool=None,
                                                   debug_mode=False,
                                                   with_cells_in_cluster_seq_sorted=False,
                                                   fct_to_keep_best_silhouettes=np.mean,
                                                   save_co_var_kmean_plot=False,
                                                   save_stat_kmean=False):
    """

    :param labels:
    :param activity_threshold:
    :param range_n_clusters_k_mean:
    :param n_surrogate_k_mean: int or list of 2 int, indicate how many surrogates should be used to determine if a
    k-mean results pass the threshold. If there are two values, then the first value indicate a low version,
    that is tried fast to save computational time, if it passed the threshold, then the second value is run for
    confirmation.
    :param k_mean_n_trials: int or list of 2 int, indicate how many trial of k-mean should be run to decide which
    cluster holds the best silhouette (the best orthogonality in a sort). It's different than finding a threshold.
    :param min_n_assemblies: if there are two number of surrogates for kmean, this is used to try the second if the
    minimum of assemblies has been found. The assemblies figure is also produce only if this given number is reached
    :param spike_nums_to_use:
    :param params_to_save_dict: (dict) params to save in the yaml file, key represent the param id, value its value
    should be simple variable type (list, int, float, str), avoid tuples
    :param cellsinpeak:
    :param data_descr:
    :param path_results:
    :param sliding_window_duration:
    :param sce_times_numbers:
    :param SCE_times:
    :param perc_threshold:
    :param n_surrogate_activity_threshold:
    :param with_shuffling:
    :param sce_times_bool:
    :param debug_mode:
    :param keep_only_the_best:
    :param with_cells_in_cluster_seq_sorted:
    :param fct_to_keep_best_silhouettes:
    :param save_co_var_kmean_plot:
    :param save_stat_kmean:
    :return:
    """
    start_time = time.time()
    results = compute_kmean(data_id=data_descr, neurons_labels=labels, cellsinpeak=cellsinpeak,
                            n_surrogates_k_mean=n_surrogate_k_mean, n_trials_k_mean=k_mean_n_trials,
                            path_results=path_results, min_n_assemblies=min_n_assemblies,
                            range_n_clusters=range_n_clusters_k_mean,
                            fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                            debug_mode=debug_mode, keep_only_the_best=keep_only_the_best,
                            sliding_window_duration=sliding_window_duration,
                            activity_threshold=activity_threshold,
                            perc_threshold_for_kmean_surrogates=perc_threshold_for_kmean_surrogates,
                            SCE_times=SCE_times)

    stop_time = time.time()
    print(f"Time to compute_kmean "
          f"{np.round(stop_time - start_time, 3)} s")

    if results is None:
        return

    range_n_clusters_k_mean, best_kmeans_by_cluster, m_cov_sces, cluster_labels_for_neurons, surrogate_percentiles, \
    significant_sce_clusters, cluster_with_best_silhouette_score, cas_dict = results

    if keep_only_the_best:
        print(f"Best number of clusters: {cluster_with_best_silhouette_score}")
        range_n_clusters_k_mean = [cluster_with_best_silhouette_score]

    # start_time_cas_plot = time.time()
    data_descr_backup = data_descr
    for n_cluster in range_n_clusters_k_mean:
        cas = cas_dict[n_cluster]

        # print(f"cas.n_cells_in_cell_assemblies_clusters {cas.n_cells_in_cell_assemblies_clusters}")
        if debug_mode:
            print(f"Results from K-means with {n_cluster} clusters: Do the plots")
        data_descr = data_descr_backup
        if n_cluster == cluster_with_best_silhouette_score:
            data_descr += "_best_cluster"

        cas.neurons_labels = labels
        cas.sliding_window_duration = sliding_window_duration

        # plot only if cell assembly exists
        if len(cas.n_cells_in_cell_assemblies_clusters) > 0:
            cas.plot_cell_assemblies(data_descr=data_descr, spike_nums=spike_nums_to_use,
                                     SCE_times=SCE_times, activity_threshold=activity_threshold,
                                     with_cells_in_cluster_seq_sorted=False,
                                     sce_times_bool=sce_times_bool,
                                     save_formats=['pdf', 'png'])  # "pdf"

        # cas.save_data_on_file(n_clusters=n_cluster)
        cas.save_data_on_yaml_file(n_clusters=n_cluster, params_to_save_dict=params_to_save_dict)
        #
        # stop_time_cas_plot = time.time()
        # print(f"Time to plot cas plot  "
        #       f"{(stop_time_cas_plot - start_time_cas_plot):3f} s")

        # TODO: see while plotting twice cell_assemblies make it bug
        if with_cells_in_cluster_seq_sorted:
            cas.plot_cell_assemblies(data_descr=data_descr + "_seq", spike_nums=spike_nums_to_use,
                                     SCE_times=SCE_times, activity_threshold=activity_threshold,
                                     with_cells_in_cluster_seq_sorted=True,
                                     sce_times_bool=sce_times_bool)

        # if at least a cell assmbly is found, we plot it.
        if save_co_var_kmean_plot or (len(cas.n_cells_in_cell_assemblies_clusters) > 0):
            # this section will order the spike_nums for display purpose
            clustered_spike_nums = np.copy(spike_nums_to_use)
            cell_labels = []
            cluster_labels = cluster_labels_for_neurons[n_cluster]
            cluster_horizontal_thresholds = []
            cells_to_highlight = []
            cells_to_highlight_colors = []
            start = 0
            for k in np.arange(-1, np.max(cluster_labels) + 1):
                e = np.equal(cluster_labels, k)
                nb_k = np.sum(e)
                if nb_k == 0:
                    continue
                cells_indices = np.where(e)[0]
                # if with_cells_in_cluster_seq_sorted and (len(cells_indices) > 2):
                #     result_ordering = order_spike_nums_by_seq(spike_nums_to_use[cells_indices, :], param,
                #                                               debug_mode=debug_mode,
                #                                               sce_times_bool=sce_times_bool)
                #     seq_dict_tmp, ordered_indices, all_best_seq = result_ordering
                #     # if a list of ordered_indices, the size of the list is equals to ne number of cells,
                #     # each list correspond to the best order with this cell as the first one in the ordered seq
                #     if ordered_indices is not None:
                #         cells_indices = cells_indices[ordered_indices]
                # clustered_spike_nums[start:start + nb_k, :] = spike_nums_to_use[cells_indices, :]
                for index in cells_indices:
                    cell_labels.append(labels[index])
                if k >= 0:
                    # color = plt.nipy_spectral(float(k + 1) / (n_cluster + 1))
                    color = BREWER_COLORS[k % len(BREWER_COLORS)]
                    cell_indices = list(np.arange(start, start + nb_k))
                    cells_to_highlight.extend(cell_indices)
                    cells_to_highlight_colors.extend([color] * len(cell_indices))
                start += nb_k
                if (k + 1) < (np.max(cluster_labels) + 1):
                    cluster_horizontal_thresholds.append(start)

            start_time_co_var = time.time()
            # figure with k-mean results
            # ######### Plot that show cluster activation
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
            if len(cell_labels) > 100:
                y_ticks_labels_size = 1
            else:
                y_ticks_labels_size = 3
            spike_shape_size = 1
            if len(cell_labels) > 150:
                spike_shape_size = 0.5
            plot_raster(spike_nums=clustered_spike_nums, path_results=path_results,
                        spike_train_format=False,
                        title=f"{n_cluster} clusters raster plot {data_descr}",
                        file_name=f"spike_nums_{data_descr}_{n_cluster}_clusters",
                        y_ticks_labels=cell_labels,
                        y_ticks_labels_size=y_ticks_labels_size,
                        save_raster=False,
                        show_raster=False,
                        plot_with_amplitude=False,
                        activity_threshold=activity_threshold,
                        raster_face_color='black',
                        cell_spikes_color='white',
                        horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                        horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                        horizontal_lines_sytle="dashed",
                        horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                        # vertical_lines=SCE_times,
                        # vertical_lines_colors=['white'] * len(SCE_times),
                        # vertical_lines_sytle="solid",
                        # vertical_lines_linewidth=[0.2] * len(SCE_times),
                        span_area_coords=[SCE_times],
                        span_area_colors=['white'],
                        cells_to_highlight=cells_to_highlight,
                        cells_to_highlight_colors=cells_to_highlight_colors,
                        sliding_window_duration=sliding_window_duration,
                        show_sum_spikes_as_percentage=True,
                        spike_shape="o",
                        spike_shape_size=spike_shape_size,
                        save_formats="pdf",
                        axes_list=[ax1, ax2],
                        SCE_times=SCE_times)

            show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
                                     significant_sce_clusters=significant_sce_clusters[n_cluster],
                                     n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
                                     cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                     data_str=data_descr, path_results=path_results,
                                     show_silhouettes=True, neurons_labels=labels,
                                     surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
                                     axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats=["pdf", "png"])
            plt.close()
            # stop_time_co_var = time.time()
            # print(f"Time to plot kmean & co_var  "
            #       f"{np.round(stop_time_co_var - start_time_co_var, 3)} s")

    # do_cluster_activations_computing(cas)
    if save_stat_kmean:
        save_stat_SCE_and_cluster_k_mean_version(spike_nums_to_use=spike_nums_to_use,
                                                 data_descr=data_descr,
                                                 activity_threshold=activity_threshold,
                                                 k_means=best_kmeans_by_cluster[n_cluster],
                                                 SCE_times=SCE_times, n_cluster=n_cluster, path_results=path_results,
                                                 sliding_window_duration=sliding_window_duration,
                                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                                 perc_threshold=perc_threshold_for_sce,
                                                 n_surrogate_k_mean=n_surrogate_k_mean,
                                                 n_surrogate_activity_threshold=n_surrogate_activity_threshold)
    # stop_time = time.time()
    # print(f"Time to compute_and_plot_clusters_raster_kmean_version "
    #       f"{np.round(stop_time - start_time, 3)} s")
