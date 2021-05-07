import logging
from handler.pygmo_fitting_report_creator import PyGMOFittingReportCreator

if __name__ == "__main__":
    # set logging level to highest level
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

    # Create report creator object and run report creation
    src = PyGMOFittingReportCreator()

    # saves fitting results in a readable json format to the data-storage folder
    src.write_data_report(clear_all=True)

    # creates the grid search overview table for the Appendix
    # src.create_latex_table()
