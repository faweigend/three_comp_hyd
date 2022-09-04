import logging

from pypermod.utility import PlotLayout
from threecomphyd.evolutionary_fitter.three_comp_tools import prepare_standard_tte_measures, \
    prepare_caen_recovery_ratios
from threecomphyd.investigate.three_comp_fitness_investigator import ThreeCompFitnessInvestigator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-5s %(name)s - %(message)s. [file=%(filename)s:%(lineno)d]")

PlotLayout.set_rc_params()

p = [18042.056916563655,
     46718.177938027344,
     247.39628102450715,
     106.77042879166977,
     16.96027055119397,
     0.715113715965181,
     0.01777338005017555,
     0.24959503053279475]

# example W' and CP settings
w_p = 18200
cp = 248

# create the energy expenditure measures to fit to
ttes = prepare_standard_tte_measures(w_p=w_p, cp=cp)

# the recovery measures to fit to
recovery_measures = prepare_caen_recovery_ratios(w_p=w_p, cp=cp)

# the investigator to visualise performance
ThreeCompFitnessInvestigator(ttes=ttes, recovery_measures=recovery_measures, load_config=p)
