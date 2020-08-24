from cicada.analysis.cicada_analysis import CicadaAnalysis
from time import time
import numpy as np
from pymeica.utils.display.distribution_plot import plot_box_plots
from pymeica.utils.display.colors import BREWER_COLORS
from sortedcontainers import SortedDict


class CicadaCaBySleepStage(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Display cell assemblies stats</b></p><br>'
        CicadaAnalysis.__init__(self, name="Cell assemblies by sleep stage", family_id="Descriptive",
                                short_description="Display cell assemblies stats",
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
        analysis_copy = CicadaCaBySleepStage(config_handler=self.config_handler)
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

        return True

    def set_arguments_for_gui(self):
        """

        Returns:

        """
        CicadaAnalysis.set_arguments_for_gui(self)

        self.add_open_dir_dialog_arg_for_gui(arg_name="mcad_data_path", mandatory=True,
                                             short_description="MCAD results data path",
                                             long_description="To get a summary of Malvache Cell Assemblies Detection "
                                                              "previsouly done for those session, indicate the path where"
                                                              "to find the yaml file containing the results.",
                                             key_names=None, family_widget="mcad")

        choices_dict = dict()
        for session_data in self._data_to_analyse:
            session_id = session_data.identifier
            self.sleep_stage_selection_to_index[session_id] = dict()
            sleep_stages = session_data.sleep_stages
            sleep_stages_description = [f"{ss.number} - stage {ss.sleep_stage} - {ss.duration_sec} sec "
                                        for ss in sleep_stages]
            for ss_index, ss in enumerate(sleep_stages_description):
                self.sleep_stage_selection_to_index[session_id][ss] = ss_index
            choices_dict[session_id] = sleep_stages_description
        self.add_choices_arg_for_gui(arg_name="sleep_stages_selected", choices=choices_dict,
                                     default_value=None,
                                     short_description=f"Sleep stages to analyse",
                                     multiple_choices=True,
                                     family_widget="sleep_stages")

        self.add_image_format_package_for_gui()

        self.add_verbose_arg_for_gui()

        self.add_with_timestamp_in_filename_arg_for_gui()


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

        mcad_data_path = kwargs.get("mcad_data_path")

        if mcad_data_path is None:
            print(f"You need to indicate the path to load MCAD data")
            self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)
            print(f"Raster analysis run in {time() - self.analysis_start_time} sec")
            return

        sleep_stages_selected_by_session = kwargs["sleep_stages_selected"]
        # print(f"sleep_stages_selected {sleep_stages_selected_by_session}")

        save_formats = kwargs["save_formats"]
        if save_formats is None:
            save_formats = "pdf"

        save_raster = True

        dpi = kwargs.get("dpi", 100)

        width_fig = kwargs.get("width_fig")

        height_fig = kwargs.get("height_fig")

        with_timestamps_in_file_name = kwargs.get("with_timestamp_in_file_name", True)

        n_sessions = len(self._data_to_analyse)

        sleep_stages_to_analyse_by_subject = dict()
        # TODO: Display the average duration of chunk containing cell assemblies in each stage
        # TODO: Show evolution with the night for each sleep stage ?
        for session_index, session_data in enumerate(self._data_to_analyse):
            session_identifier = session_data.identifier
            print(f"-------------- {session_identifier} -------------- ")
            sleep_stages_selected = sleep_stages_selected_by_session[session_identifier]
            if len(sleep_stages_selected) == 0:
                print(f"No sleep stage selected for {session_identifier}")
                sleep_stages_to_analyse_by_subject[session_identifier] = []
                continue
            sleep_stages_selected = [self.sleep_stage_selection_to_index[session_identifier][ss]
                                     for ss in sleep_stages_selected]

            session_data.load_mcad_data(data_path=mcad_data_path)
            sleep_stages_to_analyse_by_subject[session_identifier] = sleep_stages_selected
            # session_data.descriptive_stats()
            # self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)

        plot_ca_proportion_night_by_sleep_stage(subjects_data=self._data_to_analyse,
                                                sleep_stages_to_analyse_by_subject=sleep_stages_to_analyse_by_subject,
                                                results_path=self.get_results_path(),
                                                save_formats=save_formats)

        self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)
        print(f"Analysis run in {time() - self.analysis_start_time} sec")


def plot_ca_proportion_night_by_sleep_stage(subjects_data, sleep_stages_to_analyse_by_subject, results_path,
                                            save_formats):
    """

    :param subjects_data:
    :param sleep_stages_to_analyse_by_subject:
    :param results_path:
    :param save_formats:
    :return:
    """

    subject_descr = ""
    total_sleep_duration_by_stage = dict()
    # 2nd key is the number of cell assemblies
    total_time_with_ca_by_stage = dict()
    n_chunks_by_stage_and_ca = dict()
    for subject_data in subjects_data:
        subject_id = subject_data.identifier
        subject_descr = subject_descr + subject_id + "_"
        for sleep_stage_index in sleep_stages_to_analyse_by_subject[subject_id]:
            sleep_stage = subject_data.sleep_stages[sleep_stage_index]
            # sleep_stage.sleep_stage is a string representing the sleep stage like 'W' or '3'
            if sleep_stage.sleep_stage not in total_sleep_duration_by_stage:
                total_sleep_duration_by_stage[sleep_stage.sleep_stage] = 0
                n_chunks_by_stage_and_ca[sleep_stage.sleep_stage] = dict()
                total_time_with_ca_by_stage[sleep_stage.sleep_stage] = dict()

            # adding to 0 to all
            if 0 not in total_time_with_ca_by_stage[sleep_stage.sleep_stage]:
                total_time_with_ca_by_stage[sleep_stage.sleep_stage][0] = 0
                n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][0] = 0
            total_sleep_duration_by_stage[sleep_stage.sleep_stage] += sleep_stage.duration_sec
            if len(sleep_stage.mcad_outcomes) == 0:
                # then there is no cell assembly
                total_time_with_ca_by_stage[sleep_stage.sleep_stage][0] += sleep_stage.duration_sec
                # print(f"{sleep_stage_index} - {sleep_stage.sleep_stage} sleep_stage.duration_sec: {sleep_stage.duration_sec}")
                n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][0] += 1
                continue
            # print(f"{sleep_stage_index} - {sleep_stage.sleep_stage} N mcad_outcomes: {len(sleep_stage.mcad_outcomes)}")
            for bins_tuple, mcad_outcome in sleep_stage.mcad_outcomes.items():
                n_bins = bins_tuple[1] - bins_tuple[0] + 1
                duration_in_sec = (n_bins * mcad_outcome.spike_trains_bin_size) / 1000
                # we round it as bin sometimes remove times if there are no spikes
                if abs(duration_in_sec - sleep_stage.duration_sec) < 15:
                    duration_in_sec = sleep_stage.duration_sec
                # print(f"{sleep_stage_index} - {sleep_stage.sleep_stage} duration_in_sec: {duration_in_sec}")
                if mcad_outcome.n_cell_assemblies not in total_time_with_ca_by_stage[sleep_stage.sleep_stage]:
                    total_time_with_ca_by_stage[sleep_stage.sleep_stage][mcad_outcome.n_cell_assemblies] = 0
                total_time_with_ca_by_stage[sleep_stage.sleep_stage][mcad_outcome.n_cell_assemblies] += duration_in_sec

                if mcad_outcome.n_cell_assemblies not in n_chunks_by_stage_and_ca[sleep_stage.sleep_stage]:
                    n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][mcad_outcome.n_cell_assemblies] = 0
                n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][mcad_outcome.n_cell_assemblies] += 1


    # for sleep_stage, duration_in_stage in total_sleep_duration_by_stage.items():
    for sleep_stage in total_time_with_ca_by_stage.keys():
        box_plot_dict = SortedDict()
        total_duration_in_stage = 0
        for duration_in_ca in total_time_with_ca_by_stage[sleep_stage].values():
            total_duration_in_stage += duration_in_ca
        for n_cell_assemblies, duration_in_ca in total_time_with_ca_by_stage[sleep_stage].items():
            title = f"{n_cell_assemblies}\nn={n_chunks_by_stage_and_ca[sleep_stage][n_cell_assemblies]}"
            box_plot_dict[title] = [(duration_in_ca / total_duration_in_stage) * 100]
        plot_box_plots(data_dict=box_plot_dict, title="",
                       filename=f"{subject_descr}cell_ass_in_stage_{sleep_stage}",
                       with_x_jitter=False,
                       path_results=results_path, with_scatters=True,
                       scatter_size=300, link_medians=True,
                       y_label=f"Time proportion (%) over {int(total_duration_in_stage)} sec", colors=BREWER_COLORS,
                       save_formats=save_formats)


