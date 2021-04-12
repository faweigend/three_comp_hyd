## A New Pathway to Approximate Energy Expenditure andRecovery of an Athlete

![Alt Text](./httpdocs/title.gif)

### Setup And Hints

We recommend the use of a virtual environment. All that is required is to install these two dependencies via `pip3`
```
matplotlib
pygmo
```

If you use pycharm professional, please ensure that the SciView tab is deactivated when running interactive animations.
```
File => Settings => Tools => Python Scientific => Uncheck "show plots in tool window"
```

### Usage

Three demo applications are available via scripts in the base directory 
* `interactive_simulation.py` lets you experiment with an exemplary three component hydraulic agent and 
investigate its responses to various power demands.
* `model_behaviour_plots.py` recreates the energy expenditure and recovery plots of the results section of the paper.
* `fit_with_pygmo.py` uses evolutionary computation to find configurations for the three component hydraulic model that 
  makes it recreate published measures by Caen et al. (see paper). The script starts a grid search over described 
  parameter settings for MOEA/D coupled with the asynchronous island model. Results are stored into a `data-storage` 
  folder in the root directory of the project.
