from cicada.analysis.cicada_analysis import CicadaAnalysis
from time import time
import numpy as np


class CicadaDisplayPatientInfo(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Display patient information</b></p><br>'
        CicadaAnalysis.__init__(self, name="patient_info", family_id="Descriptive",
                                short_description="Display patient information",
                                long_description=long_description,
                                config_handler=config_handler,
                                accepted_data_formats=["sEEG_BONN"])

    def copy(self):
        """
        Make a copy of the analysis
        Returns:

        """
        analysis_copy = CicadaDisplayPatientInfo(config_handler=self.config_handler)
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
            if session_data.DATA_FORMAT != "sEEG_BONN":
                self.invalid_data_help = f"Non sEEG_BONN format compatibility not yet implemented: " \
                                         f"{session_data.DATA_FORMAT}"
                return False

        return True

    def set_arguments_for_gui(self):
        """

        Returns:

        """
        CicadaAnalysis.set_arguments_for_gui(self)

        self.add_bool_option_for_gui(arg_name="test_bool", true_by_default=False,
                                     short_description="Test",
                                     family_widget="test")

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

        test_bool = kwargs.get("test_bool", True)

        n_sessions = len(self._data_to_analyse)

        for session_index, session_data in enumerate(self._data_to_analyse):
            session_identifier = session_data.identifier
            print(f"-------------- {session_identifier} -------------- ")

            self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)

        print(f"Raster analysis run in {time() - self.analysis_start_time} sec")
