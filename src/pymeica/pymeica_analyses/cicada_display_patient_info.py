from cicada.analysis.cicada_analysis import CicadaAnalysis
from time import time
import numpy as np


class CicadaDisplayPatientInfo(CicadaAnalysis):
    def __init__(self, config_handler=None):
        """
        """
        long_description = '<p align="center"><b>Display subject information</b></p><br>'
        CicadaAnalysis.__init__(self, name="Subject description", family_id="Descriptive",
                                short_description="Display subject information",
                                long_description=long_description,
                                config_handler=config_handler,
                                accepted_data_formats=["PyMEICA"])

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

        self.add_open_dir_dialog_arg_for_gui(arg_name="mcad_data_path", mandatory=False,
                                             short_description="MCAD results data path",
                                        long_description="To get a summary of Malvache Cell Assemblies Detection "
                                                         "previsouly done for those session, indicate the path where"
                                                         "to find the yaml file containing the results.",
                                        key_names=None, family_widget="mcad")


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

        n_sessions = len(self._data_to_analyse)

        for session_index, session_data in enumerate(self._data_to_analyse):
            session_identifier = session_data.identifier
            print(f"-------------- {session_identifier} -------------- ")
            session_data.load_mcad_data(data_path=mcad_data_path)
            session_data.descriptive_stats()
            self.update_progressbar(time_started=self.analysis_start_time, increment_value=100 / n_sessions)

        print(f"Display patient info analysis run in {time() - self.analysis_start_time} sec")
