import logging
import os

from threecomphyd.handler.pygmo_fitting_report_creator import PyGMOFittingReportCreator

import threecomphyd.config as config

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # set the path to read the results from. The ReportCreator will look for a THREE_COMP_PYGMO_FIT directory
    config.paths["data_storage"] = os.path.dirname(os.path.abspath(__file__)) + "/../data-storage/"

    # Create report creator object
    src = PyGMOFittingReportCreator()

    # run report creation, which saves fitting results in a readable json format into the
    # config.paths["data_storage"] + "PYGMO_FITTING_REPORT_CREATOR" folder
    src.write_data_report(clear_all=True)

    # creates the grid search overview table for the Appendix of Weigend et al. 2021
    # src.create_latex_table()
