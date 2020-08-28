from cicada.analysis.cicada_analysis import CicadaAnalysis
from cicada.utils.misc import validate_indices_in_string_format, \
    extract_indices_from_string
from time import time
import numpy as np
from pymeica.utils.display.pymeica_plots import plot_scatter_family
from sortedcontainers import SortedDict


class CicadaCaRepeatOverNight(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Cell assembly repeats over night</b></p><br>'
        CicadaAnalysis.__init__(self, name="Cell assembly repeats",
                                family_id="Over night evolution",
                                short_description="Display cell assembly repeats over night",
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
        analysis_copy = CicadaCaRepeatOverNight(config_handler=self.config_handler)
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

        self.add_bool_option_for_gui(arg_name="only_ca_with_ri", true_by_default=False,
                                     short_description="Only CA with RI",
                                     long_description="Only consider cell assemblies with responsive unit",
                                     family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="min_repeat_in_ca", min_value=1, max_value=10,
                                        short_description="Min repeat in cell assembly",
                                        long_description="Only cell assemblies with this minimum repeat will be included",
                                        default_value=3, family_widget="data_params")

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

        dpi = kwargs.get("dpi", 100)

        width_fig = kwargs.get("width_fig")

        height_fig = kwargs.get("height_fig")

        with_timestamps_in_file_name = kwargs.get("with_timestamp_in_file_name", True)

        n_sessions = len(self._data_to_analyse)

        only_ca_with_ri = kwargs.get("only_ca_with_ri", False)

        min_repeat_in_ca = kwargs.get("min_repeat_in_ca", 3)

        sleep_stages_to_analyse_by_subject = dict()

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
            # print(f"sleep_stages_selected {sleep_stages_selected}")

            if side_to_analyse == "L&R":
                side_to_load = None
            else:
                side_to_load = side_to_analyse
            session_data.load_mcad_data(data_path=mcad_data_path, side_to_load=side_to_load,
                                        sleep_stage_indices_to_load=sleep_stages_selected)
            sleep_stages_to_analyse_by_subject[session_identifier] = sleep_stages_selected
            # session_data.descriptive_stats()
            # self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)

        plot_ca_repeat_over_night_by_sleep_stage(subjects_data=self._data_to_analyse,
                                                 side_to_analyse=side_to_analyse,
                                                 color_by_sleep_stage_dict=color_by_sleep_stage_dict,
                                                 sleep_stages_to_analyse_by_subject=sleep_stages_to_analyse_by_subject,
                                                 only_ca_with_ri=only_ca_with_ri,
                                                 min_repeat_in_ca=min_repeat_in_ca,
                                                 results_path=self.get_results_path(),
                                                 save_formats=save_formats)

        self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)
        print(f"Score cell assemblies over night analysis run in {time() - self.analysis_start_time:.2f} sec")


def plot_ca_repeat_over_night_by_sleep_stage(subjects_data, side_to_analyse, color_by_sleep_stage_dict,
                                             only_ca_with_ri, min_repeat_in_ca,
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
    # key is sleep_stage_name, value is a list of 2 list, first list contains the time elapsed since falling asleep,
    # second is the score of the assembly (2 list are the same length)
    ca_repeat_by_stage_dict = dict()
    time_by_stage_with_ca_dict = dict()
    time_total_by_stage_dict = dict()
    n_assemblies_by_stage_dict = dict()
    for subject_data in subjects_data:
        subject_id = subject_data.identifier
        subject_descr = subject_descr + subject_id + "_"
        if side_to_analyse == "L&R":
            sides = ['L', 'R']
        else:
            sides = [side_to_analyse]
        for side in sides:
            # print(f'plot_ca_proportion_night_by_sleep_stage side {side}')
            for sleep_stage_index in np.sort(sleep_stages_to_analyse_by_subject[subject_id]):
                sleep_stage = subject_data.sleep_stages[sleep_stage_index]
                if sleep_stage.sleep_stage not in time_total_by_stage_dict:
                    time_total_by_stage_dict[sleep_stage.sleep_stage] = 0
                    time_by_stage_with_ca_dict[sleep_stage.sleep_stage] = 0
                time_total_by_stage_dict[sleep_stage.sleep_stage] += sleep_stage.duration_sec
                elapsed_time = subject_data.elapsed_time_from_falling_asleep(sleep_stage=sleep_stage)
                if elapsed_time < 0:
                    continue
                # sleep_stage.sleep_stage is a string representing the sleep stage like 'W' or '3'

                if len(sleep_stage.mcad_outcomes) == 0:
                    # then there is no cell assembly
                    continue

                time_cover_by_bin_tuples = 0
                # to count the chunks
                current_time_in_sleep_stage = 0
                for bins_tuple, mcad_outcome in sleep_stage.mcad_outcomes.items():
                    n_bins = bins_tuple[1] - bins_tuple[0] + 1
                    chunk_duration_in_sec = (n_bins * mcad_outcome.spike_trains_bin_size) / 1000

                    if mcad_outcome.side != side:
                        # only keeping the outcome from the correct side
                        current_time_in_sleep_stage += chunk_duration_in_sec
                        continue
                    if mcad_outcome.n_cell_assemblies == 0:
                        current_time_in_sleep_stage += chunk_duration_in_sec
                        continue

                    # init
                    if sleep_stage.sleep_stage not in ca_repeat_by_stage_dict:
                        ca_repeat_by_stage_dict[sleep_stage.sleep_stage] = [[], []]
                        n_assemblies_by_stage_dict[sleep_stage.sleep_stage] = 0

                    # instances of CellAssembly
                    cell_assembly_added = False
                    for cell_assembly in mcad_outcome.cell_assemblies:
                        if only_ca_with_ri and (cell_assembly.n_responsive_units == 0):
                            continue
                        if cell_assembly.n_repeats < min_repeat_in_ca:
                            continue

                        score = cell_assembly.probability_score

                        n_assemblies_by_stage_dict[sleep_stage.sleep_stage] += 1

                        if not cell_assembly_added:
                            time_by_stage_with_ca_dict[sleep_stage.sleep_stage] += chunk_duration_in_sec
                            time_elapsed_in_sec = elapsed_time + current_time_in_sleep_stage
                            time_elapsed_in_hours = time_elapsed_in_sec / 3600
                        ca_repeat_by_stage_dict[sleep_stage.sleep_stage][0].append(time_elapsed_in_hours)
                        # n repeat normalizing by the number of minutes in the chunk
                        ca_repeat_by_stage_dict[sleep_stage.sleep_stage][1].append(cell_assembly.n_repeats /
                                                                                   (chunk_duration_in_sec / 60))
                        cell_assembly_added = True

                    current_time_in_sleep_stage += chunk_duration_in_sec

    # moving winodw to average over time the repeats
    # key is the sleep stage name, value is a list of list of list of 2 float representing (x, y) for line plots
    avg_repeat_hypno_by_stage_dict = dict()
    # in hours
    window_length = 0.5
    step_length = 0.25
    min_time = 10000
    max_time = 0
    for sleep_stage_name, scatter_values in ca_repeat_by_stage_dict.items():
        times = scatter_values[0]
        min_time = min(min_time, np.min(times))
        max_time = max(max_time, np.max(times))
    bin_edges = np.arange(min_time, max_time + step_length, step_length)

    y_pos_sep = -0.25
    y_pos_sleep_stage = y_pos_sep
    for sleep_stage_name, scatter_values in ca_repeat_by_stage_dict.items():
        avg_repeat_hypno_by_stage_dict[sleep_stage_name] = [[[], []]]
        times = scatter_values[0]
        repeats = np.asarray(scatter_values[1])
        for step_index, bin_edge in enumerate(bin_edges[:-1]):
            next_bin_edge = bin_edges[step_index+1]
            center_time = (bin_edge + next_bin_edge) / 2
            indices = np.where(np.logical_and(times >= bin_edge, times <= next_bin_edge))[0]
            avg_repeat_hypno_by_stage_dict[sleep_stage_name][0][0].append(center_time)
            if len(indices) == 0:
                avg_repeat_hypno_by_stage_dict[sleep_stage_name][0][1].append(0)
            else:
                avg_repeat = np.mean(repeats[indices])
                avg_repeat_hypno_by_stage_dict[sleep_stage_name][0][1].append(avg_repeat)

        for sleep_stage_index in np.sort(sleep_stages_to_analyse_by_subject[subject_id]):
            sleep_stage = subject_data.sleep_stages[sleep_stage_index]
            if sleep_stage.sleep_stage != sleep_stage_name:
                continue
            elapsed_time = subject_data.elapsed_time_from_falling_asleep(sleep_stage=sleep_stage)
            if elapsed_time < 0:
                continue
            new_epoch = [[elapsed_time / 3600, (elapsed_time+sleep_stage.duration_sec) / 3600],
                         [y_pos_sleep_stage, y_pos_sleep_stage]]
            avg_repeat_hypno_by_stage_dict[sleep_stage_name].append(new_epoch)
        y_pos_sleep_stage += y_pos_sep

    h_lines_y_values = [-1*np.log(0.05)]
    # for sleep_stage, duration_in_stage in total_sleep_duration_by_stage.items():
    for sleep_stage_name, scatter_values in ca_repeat_by_stage_dict.items():
        data_dict = {sleep_stage_name: scatter_values}
        ca_time_sec = time_by_stage_with_ca_dict[sleep_stage_name]
        total_time_sec = time_total_by_stage_dict[sleep_stage_name]
        ratio_ca_time_total_time = (ca_time_sec / total_time_sec) * 100
        legend = f"(x{n_assemblies_by_stage_dict[sleep_stage_name]}) " \
                 f"{(ca_time_sec / 60):.1f} min over {(total_time_sec / 60):.1f} " \
                 f"min ({ratio_ca_time_total_time:.1f} %)"
        label_to_legend_dict = {sleep_stage_name: f"{sleep_stage_name}: {legend}"}
        avg_repeat_dict = {sleep_stage_name: avg_repeat_hypno_by_stage_dict[sleep_stage_name]}
        plot_scatter_family(data_dict=data_dict,
                            label_to_legend=label_to_legend_dict,
                            colors_dict=color_by_sleep_stage_dict,
                            filename=f"{subject_descr}ca_repeat_over_night_stage_{sleep_stage_name}_{side_to_analyse}",
                            y_label=f"N repeats / min",
                            path_results=results_path,  # y_lim=[0, 100],
                            x_label="Time (hours)",
                            y_log=False,
                            h_lines_y_values=h_lines_y_values,
                            scatter_size=150,
                            scatter_alpha=0.8,
                            lines_plot_values=avg_repeat_dict,
                            background_color="black",
                            link_scatter=False,
                            labels_color="white",
                            with_x_jitter=0.05,
                            with_y_jitter=None,
                            x_labels_rotation=None,
                            save_formats=save_formats,
                            with_timestamp_in_file_name=True)

    label_to_legend_dict = dict()
    for sleep_stage_name in ca_repeat_by_stage_dict.keys():
        ca_time_sec = time_by_stage_with_ca_dict[sleep_stage_name]
        total_time_sec = time_total_by_stage_dict[sleep_stage_name]
        ratio_ca_time_total_time = (ca_time_sec / total_time_sec) * 100
        legend = f"(x{n_assemblies_by_stage_dict[sleep_stage_name]}) " \
                 f"{(ca_time_sec / 60):.1f} min over {(total_time_sec / 60):.1f} " \
                 f"min ({ratio_ca_time_total_time:.1f} %)"
        label_to_legend_dict[sleep_stage_name] = f"{sleep_stage_name}: {legend}"

    # TODO: See to add option to have a different shape for a cell assembly depending of if it contains RU
    plot_scatter_family(data_dict=ca_repeat_by_stage_dict,
                        label_to_legend=label_to_legend_dict,
                        colors_dict=color_by_sleep_stage_dict,
                        filename=f"{subject_descr}ca_repeat_over_night_stage_{side_to_analyse}",
                        y_label=f"N repeats / min",
                        path_results=results_path,  # y_lim=[0, 100],
                        x_label="Time (hours)",
                        y_log=False,
                        h_lines_y_values=h_lines_y_values,
                        scatter_size=150,
                        scatter_alpha=0.8,
                        lines_plot_values=avg_repeat_hypno_by_stage_dict,
                        background_color="black",
                        link_scatter=False,
                        labels_color="white",
                        with_x_jitter=0.05,
                        with_y_jitter=None,
                        x_labels_rotation=None,
                        save_formats=save_formats,
                        with_timestamp_in_file_name=True)
