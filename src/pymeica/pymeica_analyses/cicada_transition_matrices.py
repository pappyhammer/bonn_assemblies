from cicada.analysis.cicada_analysis import CicadaAnalysis
from cicada.utils.misc import validate_indices_in_string_format, \
    extract_indices_from_string
from time import time
import numpy as np
from pymeica.utils.display.pymeica_plots import plot_ca_param_over_night_by_sleep_stage
from sortedcontainers import SortedDict


class CicadaTransitionMatrices(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Cell assembly repeats over night</b></p><br>'
        CicadaAnalysis.__init__(self, name="CA transition matrices",
                                family_id="connectivity",
                                short_description="Display cell assemblies' transition matrix",
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
        analysis_copy = CicadaTransitionMatrices(config_handler=self.config_handler)
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
                                               short_description=f"Stage sleep indices for {session_id}",
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

        self.add_bool_option_for_gui(arg_name="with_just_responsive_units", true_by_default=True,
                                     short_description="With just responsive units",
                                     long_description="Display only the responsive units in the transition matrix. "
                                                      "If no responsive units, then the transition matrix is not "
                                                      "plotted",
                                     family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="spike_trains_binsize", min_value=0, max_value=5,
                                        short_description="Bin size (ms)",
                                        default_value=1, family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="delay_max_bw_transitions", min_value=1, max_value=50,
                                        short_description="Delay(ms)",
                                        long_description="delay between two spikes to consider they are connected",
                                        default_value=10, family_widget="data_params")

        self.add_bool_option_for_gui(arg_name="count_just_the_next_one", true_by_default=True,
                                     short_description="Count just the next one",
                                     long_description="if True only the following spike in the order or "
                                                      "firing will be counted (if more than one spike in the same bin, "
                                                      "they will all be counted). "
                                                      "It still have to be in the delay time",
                                     family_widget="data_params")

        self.add_int_values_arg_for_gui(arg_name="min_nb_of_ri", min_value=1, max_value=10,
                                        short_description="Min number of responsive units",
                                        long_description="If 'with just responsive units' is chosen, "
                                                         "only the assembly "
                                                         "with this number of responsive units will be displayed",
                                        default_value=3, family_widget="data_params")

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

        save_formats = kwargs["save_formats"]
        if save_formats is None:
            save_formats = "pdf"

        dpi = kwargs.get("dpi", 100)

        width_fig = kwargs.get("width_fig")

        height_fig = kwargs.get("height_fig")

        with_timestamps_in_file_name = kwargs.get("with_timestamp_in_file_name", True)

        n_sessions = len(self._data_to_analyse)

        with_just_responsive_units = kwargs["with_just_responsive_units"]
        count_just_the_next_one = kwargs["count_just_the_next_one"]
        delay_max_bw_transitions = kwargs["delay_max_bw_transitions"]
        min_nb_of_ri = kwargs["min_nb_of_ri"]
        spike_trains_binsize = kwargs["spike_trains_binsize"]
        if spike_trains_binsize == 0:
            # then no binning
            spike_trains_binsize = None

        min_repeat_in_ca = kwargs.get("min_repeat_in_ca", 3)


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

            if side_to_analyse == "L&R":
                sides = ['L', 'R']
            else:
                sides = [side_to_analyse]

            for sleep_stage_index in np.sort(sleep_stages_selected):
                sleep_stage = session_data.sleep_stages[sleep_stage_index]
                sleep_stage_name = sleep_stage.sleep_stage

                if len(sleep_stage.mcad_outcomes) == 0:
                    # then there is no cell assembly
                    continue

                for bins_tuple, mcad_outcome in sleep_stage.mcad_outcomes.items():
                    n_bins = bins_tuple[1] - bins_tuple[0] + 1
                    # chunk_duration_in_sec = (n_bins * mcad_outcome.spike_trains_bin_size) / 1000
                    bin_str = f"{bins_tuple[0]}-{bins_tuple[1]}"

                    if mcad_outcome.side not in sides:
                        continue
                    if mcad_outcome.n_cell_assemblies < 2:
                        continue

                    for c_a_index, cell_assembly in enumerate(mcad_outcome.cell_assemblies):
                        if with_just_responsive_units and (cell_assembly.n_responsive_units < min_nb_of_ri):
                            continue
                        if cell_assembly.n_repeats < min_repeat_in_ca:
                            continue
                        if with_just_responsive_units:
                            n_cells_in_matrix = cell_assembly.n_responsive_units
                        else:
                            n_cells_in_matrix = cell_assembly.n_units

                        plot_file_name = f"{session_identifier}_{side_to_analyse}_" \
                                         f"ss_index_{sleep_stage_index}_ss_name_{sleep_stage_name}_" \
                                         f"{bin_str}_ca_{c_a_index}_" \
                                         f"{n_cells_in_matrix}_cells"
                        print(f"{plot_file_name}: {cell_assembly.n_units} in total")

                        cell_assembly.build_transition_matrix(with_just_responsive_units=with_just_responsive_units,
                                                              count_just_the_next_one=count_just_the_next_one,
                                                              delay_max_bw_transitions=delay_max_bw_transitions,
                                                              spike_trains_binsize=spike_trains_binsize,
                                                              plot_file_name=plot_file_name,
                                                              results_path=self.get_results_path(),
                                                              save_formats=save_formats)

        self.update_progressbar(time_started=self.analysis_start_time, new_set_value=100)
        print(f"Score transition matrix analysis run in {time() - self.analysis_start_time:.2f} sec")
