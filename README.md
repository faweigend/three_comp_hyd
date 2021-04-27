## A New Pathway to Approximate Energy Expenditure and Recovery of an Athlete

![Alt Text](./httpdocs/title.gif)

### Setup And Hints

We recommend the use of a virtual environment. Simply install the package by running `pip3 install <path_to_project_root>`

Required packages are listed in `requirements.txt` and should be installed automatically:
```
matplotlib
pygmo
```

If you use pycharm professional, please ensure that the SciView tab is deactivated when running interactive animations.
```
File => Settings => Tools => Python Scientific => Uncheck "show plots in tool window"
```

### Usage

Following demo applications are available via scripts in the base directory

* `interactive_simulation.py` lets you experiment with an exemplary three component hydraulic agent and 
investigate its responses to various power demands.

* `model_behaviour_plots.py` recreates the energy expenditure and recovery plots of the results section of the paper.

* `grid_search_evolutionary_parameters.py` starts a grid search over described parameter settings for MOEA/D coupled 
  with the asynchronous island model by Pygmo. One fitting per parameter combination as summarised in Table 3 of the Appendix 
  are estimated. Results are stored into a `data-storage` folder next to the root directory of the project.
  
* `evolutionary_fitting` Runs 10 fittings on an athlete usinig the in the script defined parameters for CP, W' and recovery 
  rations by Caen et al. The script uses the published set of best fitting parameters (30 generations, 
  40 cycles, 64 population size, 7 islands) to fit the model with the outlined evolutionary computation approach 
  (see Appendix B and C). Results are stored into a `data-storage` folder next to the root directory of the project.
