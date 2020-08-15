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

        self.add_bool_option_for_gui(arg_name="use_su_and_mu", true_by_default=True,
                                     short_description="Use SU & MU",
                                     long_description="If not checked, only SU will be used",
                                     family_widget="data_params")

        self.add_bool_option_for_gui(arg_name="all_data_by_sleep_stages", true_by_default=False,
                                     short_description="Analyse all recordings one sleep stage at the time",
                                     family_widget="data_params")

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
                                     family_widget="data_params")
        # TODO: Add in long description the number of units on each side
        self.add_choices_arg_for_gui(arg_name="side_to_analyse", choices=["L", "R"],
                                     default_value="L",
                                     short_description="Side to analyse",
                                     multiple_choices=False,
                                     family_widget="data_params")

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

        n_sessions = len(self._data_to_analyse)
        session_data = self._data_to_analyse[0]
        session_data.descriptive_stats()

        verbose = 0

        session_identifier = session_data.identifier
        print(f"-------------- {session_identifier} -------------- ")

        if all_data_by_sleep_stages:
            sleep_stages_selected = np.arange(len(session_data.sleep_stages))
        # TODO: update progress bar
        # TODO: Add arguments in widgets
        for sleep_stage in sleep_stages_selected:
            if all_data_by_sleep_stages:
                sleep_stage_index = sleep_stage
            else:
                # then it's a string describing the sleep stage and we convert it in integer.
                sleep_stage_index = self.sleep_stage_selection_to_index[sleep_stage]

            spike_struct = session_data.construct_spike_structure(sleep_stage_indices=[sleep_stage_index],
                                                                  channels_starting_by=[side_to_analyse],
                                                                  keeping_only_SU=not use_su_and_mu)
            stage_descr = f"{side_to_analyse} stage {session_data.sleep_stages[sleep_stage_index].sleep_stage} " \
                          f"index {sleep_stage_index}"

            mcad_main(stage_descr=stage_descr, results_path=self.get_results_path(),
                      k_means_cluster_size=np.arange(3, 4),
                      n_surrogate_k_mean=10,
                      spike_trains_binsize=25,
                      spike_trains=spike_struct.spike_trains, cells_label=spike_struct.labels,
                      subject_id=session_identifier,
                      remove_high_firing_cells=True,
                      n_surrogate_activity_threshold=500, perc_threshold=95, verbose=verbose,
                      firing_rate_threshold=5)

        self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)

        print(f"MCAD analysis run in {time() - self.analysis_start_time} sec")
