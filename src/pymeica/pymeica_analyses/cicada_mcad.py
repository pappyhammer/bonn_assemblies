from cicada.analysis.cicada_analysis import CicadaAnalysis
from time import time
import numpy as np
from pymeica.utils.mcad import mcad_main


class CicadaMcad(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Malvache Cell Assemblies Detection</b></p><br>'
        CicadaAnalysis.__init__(self, name="MCAD", family_id="cell assemblies detection",
                                short_description="Malvache Cell Assemblies Detection",
                                long_description=long_description,
                                config_handler=config_handler,
                                accepted_data_formats=["PyMEICA"])

        # from the choice list, give the index corresponding in the sleep_stages list
        self.sleep_stage_selection_to_index = dict()

    def copy(self):
        """
        Make a copy of the analysis
        Returns:

        """
        analysis_copy = CicadaMcad(config_handler=self.config_handler)
        self.transfer_attributes_to_tabula_rasa_copy(analysis_copy=analysis_copy)

        return analysis_copy

    def check_data(self):
        """
        Check the data given one initiating the class and return True if the data given allows the analysis
        implemented, False otherwise.
        :return: a boolean
        """
        super().check_data()

        for session_index, session_data in enumerate(self._data_to_analyse):
            if session_data.DATA_FORMAT != "PyMEICA":
                self.invalid_data_help = f"Non PyMEICA format compatibility not yet implemented: " \
                                         f"{session_data.DATA_FORMAT}"
                return False
        if len(self._data_to_analyse) > 1:
            self.invalid_data_help = "No more than one subject at a time"
            return False

        return True

    def set_arguments_for_gui(self):
        """

        Returns:

        """
        CicadaAnalysis.set_arguments_for_gui(self)

        session_data = self._data_to_analyse[0]
        sleep_stages = session_data.sleep_stages
        sleep_stages_description = [f"{ss.number} - stage {ss.sleep_stage} - {ss.duration_sec} sec "
                                    for ss in sleep_stages]
        # print(f"sleep_stages_description {sleep_stages_description}")
        for ss_index, ss in enumerate(sleep_stages_description):
            self.sleep_stage_selection_to_index[ss] = ss_index
        self.add_choices_arg_for_gui(arg_name="sleep_stages_selected", choices=sleep_stages_description,
                                     default_value=sleep_stages_description[0],
                                     short_description="Sleep stages to analyse",
                                     multiple_choices=True,
                                     family_widget="sleep_stages")

        self.add_bool_option_for_gui(arg_name="use_su_and_mu", true_by_default=True,
                                     short_description="Use SU & MU",
                                     long_description="If not checked, only SU will be used",
                                     family_widget="data_params")

        self.add_bool_option_for_gui(arg_name="all_data_by_sleep_stages", true_by_default=False,
                                     short_description="Analyse all recordings one sleep stage at the time",
                                     family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="spike_trains_bin_size", min_value=1, max_value=500,
                                        short_description="Bin size for spike trains (in ms)",
                                        default_value=25, family_widget="data_params")

        # TODO: Add in long description the number of units on each side
        self.add_choices_arg_for_gui(arg_name="side_to_analyse", choices=["L", "R"],
                                     default_value="L",
                                     short_description="Side to analyse",
                                     multiple_choices=False,
                                     family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="min_n_clusters", min_value=2, max_value=20,
                                        short_description="Minimum number of cell assemblies",
                                        long_description="A range from min to max number of cell assemblies "
                                                         "will be explored. If min == max, then one number of cell "
                                                         "assembly will be used.",
                                        default_value=2, family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="max_n_clusters", min_value=2, max_value=20,
                                        short_description="Maximum number of cell assemblies",
                                        default_value=4, family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="n_surrogates_k_mean_1st_try", min_value=5, max_value=100,
                                        short_description="N surrogates for kmean on 1st try",
                                        default_value=10, family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="n_surrogates_k_mean_2nd_try", min_value=5, max_value=1000,
                                        short_description="N surrogates for kmean on 2nd try",
                                        default_value=40, family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="k_mean_n_trials_1st_try", min_value=2, max_value=100,
                                        short_description="N trials for kmean on 1st try",
                                        default_value=5, family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="k_mean_n_trials_2nd_try", min_value=2, max_value=200,
                                        short_description="N trials for kmean on 2nd try",
                                        default_value=25, family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="perc_threshold_for_kmean_surrogates", min_value=90, max_value=99,
                                        short_description="Percentile threshold for kmean surrogate silhouettes",
                                        default_value=95, family_widget="kmeans_params")


        self.add_bool_option_for_gui(arg_name="apply_two_steps_k_mean", true_by_default=True,
                                     short_description="Apply kmeans in two steps",
                                     long_description="If True, a first set of parameter is applied to kmeans, "
                                                      "if significant clusters are found, then a second set is applied "
                                                      "and this results is kept. The second parameters "
                                                      "should be higher. "
                                                      "The idea is to save computational time. If false, the 2nd try "
                                                      "values are the ones used.",
                                     family_widget="kmeans_params")

        self.add_int_values_arg_for_gui(arg_name="n_surrogate_activity_threshold", min_value=100, max_value=5000,
                                        short_description="N surrogates to compute synchronous activity threshold",
                                        default_value=500, family_widget="se_params")

        self.add_int_values_arg_for_gui(arg_name="perc_threshold_for_sce", min_value=90, max_value=99,
                                        short_description="Percentile threshold for synchronous activity surrogates",
                                        default_value=95, family_widget="se_params")

        self.add_int_values_arg_for_gui(arg_name="min_activity_threshold", min_value=2, max_value=10,
                                        short_description="Min nb of units in synchronous event",
                                        default_value=2, family_widget="se_params")

        self.add_bool_option_for_gui(arg_name="remove_high_firing_cells", true_by_default=True,
                                     short_description="Apply firing rate threshold",
                                     long_description="Remove the cells that fire above the given threshold",
                                     family_widget="firing_rate")

        self.add_int_values_arg_for_gui(arg_name="firing_rate_threshold", min_value=1, max_value=50,
                                        short_description="Firing rate (Hz) threshold",
                                        default_value=5, family_widget="firing_rate")

        self.add_int_values_arg_for_gui(arg_name="max_size_chunk_in_sec", min_value=30, max_value=360,
                                        short_description="Max chunk of spike trains in sec to process",
                                        default_value=120, family_widget="chunk_size")

        self.add_int_values_arg_for_gui(arg_name="min_size_chunk_in_sec", min_value=10, max_value=60,
                                        short_description="Min chunk of spike trains in sec to process",
                                        long_description="Applies only for chunk that are cut, meaning shorter "
                                                         "than the max chunk size. Then if the last chunk is inferior "
                                                         "the min value, then we extend the other chunk.",
                                        default_value=40, family_widget="chunk_size")

    def update_original_data(self):
        """
        To be called if the data to analyse should be updated after the analysis has been run.
        :return: boolean: return True if the data has been modified
        """
        pass

    def run_analysis(self, **kwargs):
        """
        test
        :param kwargs:
          segmentation

        :return:
        """
        CicadaAnalysis.run_analysis(self, **kwargs)

        use_su_and_mu = kwargs.get("use_su_and_mu", True)

        all_data_by_sleep_stages = kwargs.get("all_data_by_sleep_stages", False)

        sleep_stages_selected = kwargs["sleep_stages_selected"]

        side_to_analyse = kwargs["side_to_analyse"]

        spike_trains_bin_size = kwargs.get("spike_trains_bin_size", 25)

        min_n_clusters = kwargs.get("min_n_clusters", 2)
        max_n_clusters = max(min_n_clusters, kwargs.get("max_n_clusters", 2))
        k_means_cluster_size = np.arange(min_n_clusters, max_n_clusters + 1)

        n_surrogate_activity_threshold = kwargs.get("n_surrogate_activity_threshold", 500)
        perc_threshold_for_sce = kwargs.get("perc_threshold_for_sce", 95)

        remove_high_firing_cells = kwargs.get("remove_high_firing_cells", True)
        firing_rate_threshold = kwargs.get("firing_rate_threshold", 5)

        apply_two_steps_k_mean = kwargs.get("apply_two_steps_k_mean", True)

        n_surrogates_k_mean_1st_try = kwargs.get("n_surrogates_k_mean_1st_try", 10)
        n_surrogates_k_mean_2nd_try = kwargs.get("n_surrogates_k_mean_2nd_try", 40)

        k_mean_n_trials_1st_try = kwargs.get("k_mean_n_trials_1st_try", 5)
        k_mean_n_trials_2nd_try = kwargs.get("k_mean_n_trials_2nd_try", 25)

        perc_threshold_for_kmean_surrogates = kwargs.get("perc_threshold_for_kmean_surrogates", 95)

        min_activity_threshold = kwargs.get("min_activity_threshold", 2)

        if apply_two_steps_k_mean:
            n_surrogates_k_mean = (n_surrogates_k_mean_1st_try, n_surrogates_k_mean_2nd_try)
            k_mean_n_trials = (k_mean_n_trials_1st_try, k_mean_n_trials_2nd_try)
        else:
            n_surrogates_k_mean = n_surrogates_k_mean_2nd_try
            k_mean_n_trials = k_mean_n_trials_2nd_try

        # in second, used to split the sleep stages in chunks
        max_size_chunk_in_sec = kwargs.get("max_size_chunk_in_sec", 120)
        min_size_chunk_in_sec = kwargs.get("max_size_chunk_in_sec", 40)

        n_sessions = len(self._data_to_analyse)
        session_data = self._data_to_analyse[0]
        session_data.descriptive_stats()

        verbose = 0

        session_identifier = session_data.identifier
        print("-" * 50)
        print(f"----------------- {session_identifier} ----------------- ")
        print("-" * 50)

        if all_data_by_sleep_stages:
            sleep_stages_selected = np.arange(len(session_data.sleep_stages))

        # in sec
        total_time_over_stages = 0
        # compute how much time is in all sleep stage so we update the progress bar accordingly
        for sleep_stage in sleep_stages_selected:
            if all_data_by_sleep_stages:
                sleep_stage_index = sleep_stage
            else:
                # then it's a string describing the sleep stage and we convert it in integer.
                sleep_stage_index = self.sleep_stage_selection_to_index[sleep_stage]
            total_time_over_stages += session_data.sleep_stages[sleep_stage_index].duration_sec

        for sleep_stage in sleep_stages_selected:
            if all_data_by_sleep_stages:
                sleep_stage_index = sleep_stage
            else:
                # then it's a string describing the sleep stage and we convert it in integer.
                sleep_stage_index = self.sleep_stage_selection_to_index[sleep_stage]
            print(" ")
            print("-" * 50)
            print(f"Cicada MCAD: {side_to_analyse} side, sleep stage index {sleep_stage_index}, "
                  f"{session_data.sleep_stages[sleep_stage_index].duration_sec} sec")
            print("-" * 50)

            spike_trains, spike_nums, cells_label = \
                session_data.build_spike_nums(sleep_stage_index=sleep_stage_index,
                                              side_to_analyse=side_to_analyse,
                                              keeping_only_SU=not use_su_and_mu,
                                              remove_high_firing_cells=remove_high_firing_cells,
                                              firing_rate_threshold=firing_rate_threshold,
                                              spike_trains_binsize=spike_trains_bin_size)
            # spike_struct = session_data.construct_spike_structure(sleep_stage_indices=[sleep_stage_index],
            #                                                       channels_starting_by=[side_to_analyse],
            #                                                       keeping_only_SU=not use_su_and_mu)
            # stage_descr = f"{side_to_analyse} stage {session_data.sleep_stages[sleep_stage_index].sleep_stage} " \
            #               f"index {sleep_stage_index}"
            stage_descr = f"{side_to_analyse} index {sleep_stage_index} " \
                          f"stage {session_data.sleep_stages[sleep_stage_index].sleep_stage}"

            # params to save in the yaml file
            params_to_save_dict = dict()
            params_to_save_dict["subject_id"] = session_identifier
            params_to_save_dict["spike_trains_bin_size"] = spike_trains_bin_size
            params_to_save_dict["side"] = side_to_analyse
            params_to_save_dict["sleep_stage_name"] = str(session_data.sleep_stages[sleep_stage_index].sleep_stage)
            params_to_save_dict["sleep_stage_index"] = sleep_stage_index
            params_to_save_dict["with_only_SU"] = bool(not use_su_and_mu)

            if remove_high_firing_cells:
                params_to_save_dict["firing_rate_threshold"] = int(firing_rate_threshold)

            # TODO: Add arguments in widgets
            mcad_main(stage_descr=stage_descr, results_path=self.get_results_path(),
                      k_means_cluster_size=k_means_cluster_size,
                      n_surrogates_k_mean=n_surrogates_k_mean,
                      k_mean_n_trials=k_mean_n_trials,
                      spike_trains_binsize=spike_trains_bin_size,
                      max_size_chunk_in_sec=max_size_chunk_in_sec,
                      min_size_chunk_in_sec=min_size_chunk_in_sec,
                      spike_trains=spike_trains,
                      cells_label=cells_label,
                      spike_nums=spike_nums,
                      subject_id=session_identifier,
                      min_activity_threshold=min_activity_threshold,
                      params_to_save_dict=params_to_save_dict,
                      n_surrogate_activity_threshold=n_surrogate_activity_threshold,
                      perc_threshold_for_sce=perc_threshold_for_sce,
                      verbose=verbose,
                      perc_threshold_for_kmean_surrogates=perc_threshold_for_kmean_surrogates)

            self.update_progressbar(time_started=self.analysis_start_time,
                                    increment_value=
                                    (session_data.sleep_stages[sleep_stage_index].duration_sec /
                                     total_time_over_stages) * 100)

        self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)

        print(f"MCAD analysis run in {time() - self.analysis_start_time:.2f} sec")
