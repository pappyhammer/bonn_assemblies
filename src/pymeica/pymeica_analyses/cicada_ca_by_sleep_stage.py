from cicada.analysis.cicada_analysis import CicadaAnalysis
from time import time
import numpy as np
from pymeica.utils.display.distribution_plot import plot_box_plots
from pymeica.utils.display.pymeica_plots import plot_scatter_family
from pymeica.utils.display.colors import BREWER_COLORS
from sortedcontainers import SortedDict
from cicada.utils.misc import validate_indices_in_string_format, \
    extract_indices_from_string


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
        self.stages_name = ["W", "1", "2", "3", "R"]

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

        for session_data in self._data_to_analyse:
            session_id = session_data.identifier
            self.add_field_text_option_for_gui(arg_name=f"sleep_stages_selected_by_text_{session_id}",
                                               default_value="",
                                               short_description="Stage sleep indices",
                                               long_description="You can indicate the "
                                                                "sleep stages indices in text field, "
                                                                "such as '1-4 6 15-17'to "
                                                                "make a group "
                                                                "with stages 1 to 4, 6 and 15 to 17.",
                                               family_widget="sleep_stages")

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

        side_choices = dict()
        for session_data in self._data_to_analyse:
            side_choices[session_data] = ["L", "R", "L&R"]
        self.add_choices_arg_for_gui(arg_name="side_to_analyse", choices=["L", "R", "L&R"],
                                     default_value="R",
                                     short_description="Side to analyse",
                                     multiple_choices=False,
                                     family_widget="data_params")


        for stage_name in self.stages_name:
            self.add_color_arg_for_gui(arg_name=f"color_by_stage_{stage_name}",
                                       default_value=(1, 1, 1, 1.),
                                       short_description=f"Color for stage {stage_name}",
                                       long_description=None, family_widget="stage_color")

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

        side_to_analyse = kwargs["side_to_analyse"]

        color_by_sleep_stage_dict = dict()
        for stage_name in self.stages_name:
            stage_color = kwargs[f"color_by_stage_{stage_name}"]
            color_by_sleep_stage_dict[stage_name] = stage_color

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
            sleep_stage_text_selection = kwargs.get(f"sleep_stages_selected_by_text_{session_identifier}", "")
            sleep_stages_selected = []
            if validate_indices_in_string_format(sleep_stage_text_selection):
                sleep_stages_selected = extract_indices_from_string(sleep_stage_text_selection)
            if len(sleep_stages_selected) == 0:
                sleep_stages_selected = sleep_stages_selected_by_session[session_identifier]
                sleep_stages_selected = [self.sleep_stage_selection_to_index[session_identifier][ss]
                                         for ss in sleep_stages_selected]
            if len(sleep_stages_selected) == 0:
                print(f"No sleep stage selected for {session_identifier}")
                sleep_stages_to_analyse_by_subject[session_identifier] = []
                continue

            if side_to_analyse == "L&R":
                side_to_load = None
            else:
                side_to_load = side_to_analyse
            session_data.load_mcad_data(data_path=mcad_data_path, side_to_load=side_to_load)
            sleep_stages_to_analyse_by_subject[session_identifier] = sleep_stages_selected
            # session_data.descriptive_stats()
            # self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)

        plot_ca_proportion_night_by_sleep_stage(subjects_data=self._data_to_analyse,
                                                side_to_analyse=side_to_analyse,
                                                color_by_sleep_stage_dict=color_by_sleep_stage_dict,
                                                sleep_stages_to_analyse_by_subject=sleep_stages_to_analyse_by_subject,
                                                results_path=self.get_results_path(),
                                                save_formats=save_formats)

        self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)
        print(f"Analysis run in {time() - self.analysis_start_time} sec")


def plot_ca_proportion_night_by_sleep_stage(subjects_data, side_to_analyse, color_by_sleep_stage_dict,
                                            sleep_stages_to_analyse_by_subject, results_path,
                                            save_formats):
    """

    :param subjects_data:
    :param side_to_analyse: (str)value 'L', 'R', or 'L&R'
    :param sleep_stages_to_analyse_by_subject:
    :param results_path:
    :param save_formats:
    :return:
    """
    # print("plot_ca_proportion_night_by_sleep_stage")

    subject_descr = ""
    total_sleep_duration_by_stage = dict()
    # 2nd key is the number of cell assemblies
    total_time_with_ca_by_stage = dict()
    n_chunks_by_stage_and_ca = dict()
    for subject_data in subjects_data:
        subject_id = subject_data.identifier
        subject_descr = subject_descr + subject_id + "_"
        if side_to_analyse == "L&R":
            sides = ['L', 'R']
        else:
            sides = [side_to_analyse]
        for side in sides:
            # print(f'plot_ca_proportion_night_by_sleep_stage side {side}')
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
                    n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][0] += 1
                    continue

                time_cover_by_bin_tuples = 0
                for bins_tuple, mcad_outcome in sleep_stage.mcad_outcomes.items():
                    if mcad_outcome.side != side:

                        # only keeping the outcome from the correct side
                        continue
                    n_bins = bins_tuple[1] - bins_tuple[0] + 1
                    duration_in_sec = (n_bins * mcad_outcome.spike_trains_bin_size) / 1000
                    # we round it as bin sometimes remove times if there are no spikes
                    if abs(duration_in_sec - sleep_stage.duration_sec) < 15:
                        duration_in_sec = sleep_stage.duration_sec
                    time_cover_by_bin_tuples += duration_in_sec
                    n_cell_assemblies = mcad_outcome.n_cell_assemblies
                    n_cell_assemblies = 0 if n_cell_assemblies == 1 else n_cell_assemblies
                    if n_cell_assemblies not in total_time_with_ca_by_stage[sleep_stage.sleep_stage]:
                        total_time_with_ca_by_stage[sleep_stage.sleep_stage][n_cell_assemblies] = 0
                    total_time_with_ca_by_stage[sleep_stage.sleep_stage][n_cell_assemblies] += duration_in_sec

                    if n_cell_assemblies not in n_chunks_by_stage_and_ca[sleep_stage.sleep_stage]:
                        n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][n_cell_assemblies] = 0
                    n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][n_cell_assemblies] += 1

                # in case some chunks will give no MCADOutcome
                if abs(time_cover_by_bin_tuples - sleep_stage.duration_sec) > 20:
                    n_chunks_by_stage_and_ca[sleep_stage.sleep_stage][0] += max(1, (abs(time_cover_by_bin_tuples -
                                                                                        sleep_stage.duration_sec) // 120))
                    total_time_with_ca_by_stage[sleep_stage.sleep_stage][0] += abs(time_cover_by_bin_tuples -
                                                                                   sleep_stage.duration_sec)

    # for sleep_stage, duration_in_stage in total_sleep_duration_by_stage.items():
    scatter_data_dict = SortedDict()
    for sleep_stage in total_time_with_ca_by_stage.keys():
        scatter_data_dict[sleep_stage] = [[], [], []]
        box_plot_dict = SortedDict()
        total_duration_in_stage = 0
        for duration_in_ca in total_time_with_ca_by_stage[sleep_stage].values():
            total_duration_in_stage += duration_in_ca
        # in case no cell assemblies would happen in this stage, in particular the 0 cell assembly is always added
        if total_duration_in_stage == 0:
            total_duration_in_stage = total_sleep_duration_by_stage[sleep_stage]
        for n_cell_assemblies, duration_in_ca in total_time_with_ca_by_stage[sleep_stage].items():
            title = f"{n_cell_assemblies}\nn={n_chunks_by_stage_and_ca[sleep_stage][n_cell_assemblies]}"
            time_proportion = (duration_in_ca / total_duration_in_stage) * 100
            box_plot_dict[title] = [time_proportion]
            scatter_data_dict[sleep_stage][0].append(n_cell_assemblies)
            scatter_data_dict[sleep_stage][1].append(time_proportion)
            scatter_data_dict[sleep_stage][2].append(n_chunks_by_stage_and_ca[sleep_stage][n_cell_assemblies])
        plot_box_plots(data_dict=box_plot_dict, title="",
                       filename=f"{subject_descr}cell_ass_in_stage_{sleep_stage}_{side_to_analyse}",
                       with_x_jitter=False,
                       path_results=results_path, with_scatters=True,
                       scatter_size=300, link_medians=True,
                       y_label=f"Time proportion (%) over {int(total_duration_in_stage)} sec", colors=BREWER_COLORS,
                       save_formats=save_formats)
    # print(f"scatter_data_dict {scatter_data_dict}")

    scatter_data_dict_sorted = SortedDict()
    for sleep_stage, scatter_data in scatter_data_dict.items():
        scatter_data_dict_sorted[sleep_stage] = []
        arg_sort = np.argsort(scatter_data[0])
        scatter_data_dict_sorted[sleep_stage].append(np.asarray(scatter_data[0])[arg_sort])
        scatter_data_dict_sorted[sleep_stage].append(np.asarray(scatter_data[1])[arg_sort])
        scatter_data_dict_sorted[sleep_stage].append(np.asarray(scatter_data[2])[arg_sort])

    label_to_legend_dict = dict()
    for sleep_stage in scatter_data_dict.keys():
        total_duration_in_stage = 0
        for duration_in_ca in total_time_with_ca_by_stage[sleep_stage].values():
            total_duration_in_stage += duration_in_ca
        # in case no cell assemblies would happen in this stage, in particular the 0 cell assembly is always added
        if total_duration_in_stage == 0:
            total_duration_in_stage = total_sleep_duration_by_stage[sleep_stage]
        label_to_legend_dict[sleep_stage] = f"{sleep_stage}: {total_duration_in_stage/60:.1f} min"

    plot_scatter_family(data_dict=scatter_data_dict_sorted,
                        label_to_legend=label_to_legend_dict,
                        colors_dict=color_by_sleep_stage_dict,
                        filename=f"{subject_descr}cell_ass_over_stages_{side_to_analyse}",
                        y_label=f"Time proportion (%)",
                        path_results=results_path, # y_lim=[0, 100],
                        x_label="N cells in cell assemblies",
                        y_log=False,
                        scatter_size=300,
                        scatter_alpha=1,
                        background_color="black",
                        link_scatter=True,
                        labels_color="white",
                        with_x_jitter=0.1,
                        with_text=False,
                        with_y_jitter=None,
                        x_labels_rotation=None,
                        save_formats=save_formats,
                        with_timestamp_in_file_name=True)
