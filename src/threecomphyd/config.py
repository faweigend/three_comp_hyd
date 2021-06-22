import os

# paths to be used by all scripts that store or access data
paths = {
    "data_storage": os.path.dirname(os.path.abspath(__file__)) + "/../../data-storage/"
}

# an additional constraint on the three component hydraulic model that limits the interval for phi
three_comp_phi_constraint = False
