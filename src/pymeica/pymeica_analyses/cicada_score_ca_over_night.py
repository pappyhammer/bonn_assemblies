from cicada.analysis.cicada_analysis import CicadaAnalysis
from cicada.utils.misc import validate_indices_in_string_format, \
    extract_indices_from_string
from time import time
import numpy as np
from pymeica.utils.display.pymeica_plots import plot_ca_param_over_night_by_sleep_stage
from sortedcontainers import SortedDict


class CicadaScoreCaOverNight(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Cell assembly score over night</b></p><br>'
        CicadaAnalysis.__init__(self, name="Cell assembly score",
                                family_id="Over night evolution",
                                short_description="Display cell assembly score over night",
                                long_description=long_description,
                                config_handler=config_handler,
                                accepted_data_formats=["PyMEICA"])

        # from the choice list, give the index corresponding in the sleep_stages list
        self.sleep_stage_selection_to_index = dict()
        self.stages_name = ["W", "1", "2", "3", "R"]
        # used for figure legend
        self.marker_to_brain_region = {'o': 'Amygdala', 's': 'Hippocampus',
                                       '*': "Entorhinal Cortex", 'v': 'Parahippocampal Cortex'}
        self.brain_region_to_marker = {'A': 'o', 'H': 's', 'AH': 's', 'MH': 's', 'PH': 's', 'EC': "*", 'PHC': 'v'}

    def copy(self):
        """
        Make a copy of the analysis
        Returns:

        """
        analysis_copy = CicadaScoreCaOverNight(config_handler=self.config_handler)
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
                                               short_description=f"Stage sleep indices {session_id}",
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

        self.add_choices_arg_for_gui(arg_name="text_in_scatter", choices=["No text", "n RU", "ratio RU / units"],
                                     default_value="n RU",
                                     short_description="Text to display in cell assembly",
                                     long_description="Fro each scatter representing a cell assembly, you can display "
                                                      "information regarding responsive units (RU) in it.",
                                     multiple_choices=False,
                                     family_widget="data_params")

        self.add_bool_option_for_gui(arg_name="only_ca_with_ri", true_by_default=False,
                                     short_description="Only CA with RI",
                                     long_description="Only consider cell assemblies with responsive unit",
                                     family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="min_repeat_in_ca", min_value=1, max_value=10,
                                        short_description="Min repeat in cell assembly",
                                        long_description="Only cell assemblies with this "
                                                         "minimum repeat will be included",
                                        default_value=3, family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="min_n_cells_assemblies", min_value=1, max_value=4,
                                        short_description="Min number of cell assemblies in a chunk of data",
                                        long_description="Only cell assemblies within chunk with "
                                                         "this minimum n cell assemblies will be included",
                                        default_value=2, family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="min_cells_in_cell_assemblies", min_value=2, max_value=10,
                                        short_description="Min number of cells in cell assembly",
                                        long_description="Only cell assemblies with "
                                                         "this minimum n cell will be included",
                                        default_value=2, family_widget="data_params")

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

        text_in_scatter = kwargs["text_in_scatter"]

        with_text = text_in_scatter != "No text"

        n_ru_in_text = text_in_scatter == "n RU"

        ration_ru_in_text = text_in_scatter == "ratio RU / units"

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

        min_n_cells_assemblies = kwargs.get("min_n_cells_assemblies", 2)

        min_cells_in_cell_assemblies = kwargs["min_cells_in_cell_assemblies"]

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
                                        sleep_stage_indices_to_load=sleep_stages_selected,
                                        update_progress_bar_fct=self.update_progressbar,
                                        time_started=self.analysis_start_time,
                                        total_increment=95 / n_sessions)
            sleep_stages_to_analyse_by_subject[session_identifier] = sleep_stages_selected
            # session_data.descriptive_stats()
            # self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)
        plot_ca_param_over_night_by_sleep_stage(subjects_data=self._data_to_analyse,
                                                side_to_analyse=side_to_analyse,
                                                param_name="score",
                                                fct_to_get_param=lambda ca: -1 * np.log(ca.probability_score),
                                                y_axis_label=f"Score  (-log(p))",
                                                color_by_sleep_stage_dict=color_by_sleep_stage_dict,
                                                sleep_stages_to_analyse_by_subject=sleep_stages_to_analyse_by_subject,
                                                only_ca_with_ri=only_ca_with_ri,
                                                with_text=with_text,
                                                n_ru_in_text=n_ru_in_text,
                                                ratio_ru_in_text=ration_ru_in_text,
                                                min_repeat_in_ca=min_repeat_in_ca,
                                                min_n_cells_assemblies=min_n_cells_assemblies,
                                                min_cells_in_cell_assemblies=min_cells_in_cell_assemblies,
                                                brain_region_to_marker=self.brain_region_to_marker,
                                                marker_to_brain_region=self.marker_to_brain_region,
                                                results_path=self.get_results_path(),
                                                save_formats=save_formats, dpi=dpi,
                                                with_mean_lines=True,
                                                h_lines_y_values=[-1 * np.log(0.05)])

        self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)
        print(f"Score cell assemblies over night analysis run in {time() - self.analysis_start_time:.2f} sec")
