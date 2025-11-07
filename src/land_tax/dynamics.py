##########################################################################
################ Solve dynamics with unconstrained land ##################
##########################################################################

# The codes here compute the initial and terminal steady states and store them in Json
# The dynamics are solved in Julia script "dynamics.jl".

try:  # pragma: no cover - fallback for direct execution
    from .model_solver import *
    from .paths import DYNAMICS_DIR, PARAMS_DIR
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

    from land_tax.model_solver import *  # type: ignore
    from land_tax.paths import DYNAMICS_DIR, PARAMS_DIR  # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

DYNAMICS_DIR.mkdir(parents=True, exist_ok=True)


# set parameters
taxes_calib = {         # benchmark tax to calibrate on
    'tau_s_inv':        0.0,
    'tau_S_inv':        0.0,
    'tau_K_inv':        0.0,
    'tau_K':            0.011,
    'tau_l_surface':    0.0,
    'tau_l_value':      0.0,
    'tau_L_surface':    0.0,
    'tau_L_value':      0.0,
    'tau_LR_surface':   0.0,
    'tau_LR_value':     0.0,
    'tau_LDV_sale':     0.0,
    'tau_LR_sale':      0.0,
    'tau_H':            0.0,
    'tau_HI':           0.0,
    'tau_D_K':          0.0,
    'tau_D_S':          0.0,
    'tau_D_s':          0.0,
    'tau_D_H':          0.0,
    'tau_D_h':          0.0,
    'tau_D_L':          0.0,
    'tau_D_l':          0.0,
    'tau_D_LR':         0.0,
    'tau_inher_K':      0.0,
    'tau_inher_S':      0.0,
    'tau_inher_s':      0.0,
    'tau_inher_H':      0.0,
    'tau_inher_h':      0.0,
    'tau_inher_L':      0.0,
    'tau_inher_l':      0.0,
    'tau_inher_LR':     0.0,
}

# the all tax set to zero case
taxes_eq0 = taxes_calib.copy()
taxes_eq0['tau_K_inv'] = 0.0
taxes_eq0['tau_K'] = 0.0

# fixed parameters
with open(PARAMS_DIR / 'ParamDynamics.json') as file:
    params = json.load(file)
    
params_exog = {key: params[key] for key in [
    "aleph", "n", "beta", "omega", "varrho", "gamma_C", "gamma_R", 
    "delta_K", "delta_S", "alpha_CH", "sigma", "a_L", "alpha_K", 
    "A", "rho_h", "rho_H"
]}

params_endog = {key: params[key] for key in ["cost_L1", "cost_L2", "R_R", "Lbar"]}

# with open('ParamExog.json') as file:
#     params_exog = json.load(file)

# with open('ParamEndog_uncons_calib.json') as file:
#     params_endog = json.load(file)


# init guess
init_guess = {              # initial guess of the x vector
    "c": 0.7501029863335733, 
    "C": 1.2273556513042563, 
    "C_R": 1.7454592520616932, 
    "s": 1.228191424105679, 
    "S": 2.888839715397431,
    "l": 0.5170629868792102, 
    "L": 1.2161883420952917, 
    "LR": 6.135219433586456, 
}

sol_taxeq0 = solve_ss(params_exog, params_endog, taxes_eq0, init_guess)


################# Experiment 1: A negative shock in initial capital stock #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt1 = json.load(file)
#     terminal_ss_exprmt1 = init_ss_exprmt1.copy()
init_ss_exprmt1 = sol_taxeq0.copy()
terminal_ss_exprmt1 = sol_taxeq0.copy()
    

init_ss_exprmt1['K'] = 0.8 * init_ss_exprmt1['K']       # negative shock

# save
with open(DYNAMICS_DIR / 'exprmt1_init.json', 'w') as file:
    json.dump(init_ss_exprmt1, file)

with open(DYNAMICS_DIR / 'exprmt1_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt1, file)
    


################# Experiment 2: A sudden jump in tax on capital #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt2 = json.load(file)
init_ss_exprmt2 = sol_taxeq0.copy()


# with open('sol_uncons_calib.json') as file:
#     terminal_ss_exprmt2 = json.load(file)
taxes_exprmt2_end = taxes_eq0.copy()
taxes_exprmt2_end['tau_K'] = taxes_calib['tau_K']
terminal_ss_exprmt2 = solve_ss(params_exog, params_endog, taxes_exprmt2_end, init_guess)


# save
with open(DYNAMICS_DIR / 'exprmt2_init.json', 'w') as file:
    json.dump(init_ss_exprmt2, file)

with open(DYNAMICS_DIR / 'exprmt2_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt2, file)




################# Experiment 3: A sudden jump in uniform tax on land surface #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt3 = json.load(file)
init_ss_exprmt3 = sol_taxeq0.copy()

# save
with open(DYNAMICS_DIR / 'exprmt3_init.json', 'w') as file:
    json.dump(init_ss_exprmt3, file)


# compute steady state 2
taxes_exprmt3_end = taxes_eq0.copy()
taxes_exprmt3_end['tau_L_surface'] = 0.02
taxes_exprmt3_end['tau_l_surface'] = 0.02
taxes_exprmt3_end['tau_LR_surface'] = 0.02

terminal_ss_exprmt3 = solve_ss(params_exog, params_endog, taxes_exprmt3_end, init_guess)

# save
with open(DYNAMICS_DIR / 'exprmt3_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt3, file)



################# Experiment 4: A geometric increase in uniform tax on land surface #################

# same initial and terminal condition as before
# save
with open(DYNAMICS_DIR / 'exprmt4_init.json', 'w') as file:
    json.dump(init_ss_exprmt3, file)

# save
with open(DYNAMICS_DIR / 'exprmt4_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt3, file)
    
    
    
################# Experiment 5: A linear increase in uniform tax on land surface #################

# same initial and terminal condition as before
# save
with open(DYNAMICS_DIR / 'exprmt5_init.json', 'w') as file:
    json.dump(init_ss_exprmt3, file)

# save
with open(DYNAMICS_DIR / 'exprmt5_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt3, file)
    
    
    
################# Experiment 6: A linear increase in uniform tax on land surface #################

# same initial and terminal condition as before
# save
with open(DYNAMICS_DIR / 'exprmt6_init.json', 'w') as file:
    json.dump(init_ss_exprmt3, file)

# save
with open(DYNAMICS_DIR / 'exprmt6_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt3, file)
    


################# Experiment 7: A sudden jump in uniform (rate) tax on land value #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt7 = json.load(file)
init_ss_exprmt7 = sol_taxeq0.copy()

# save
with open(DYNAMICS_DIR / 'exprmt7_init.json', 'w') as file:
    json.dump(init_ss_exprmt7, file)


# compute steady state 2
taxes_exprmt7_end = taxes_eq0.copy()
taxes_exprmt7_end['tau_L_value'] = 0.02
taxes_exprmt7_end['tau_l_value'] = 0.02
taxes_exprmt7_end['tau_LR_value'] = 0.02

terminal_ss_exprmt7 = solve_ss(params_exog, params_endog, taxes_exprmt7_end, init_guess)

# save
with open(DYNAMICS_DIR / 'exprmt7_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt7, file)



################# Experiment 8: A geometric increase in uniform (rate) tax on land value #################

# same initial and terminal condition as before
# save
with open(DYNAMICS_DIR / 'exprmt8_init.json', 'w') as file:
    json.dump(init_ss_exprmt7, file)

# save
with open(DYNAMICS_DIR / 'exprmt8_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt7, file)
    
    
    
################# Experiment 9: A linear increase in uniform (rate) tax on land value #################

# same initial and terminal condition as before
# save
with open(DYNAMICS_DIR / 'exprmt9_init.json', 'w') as file:
    json.dump(init_ss_exprmt7, file)

# save
with open(DYNAMICS_DIR / 'exprmt9_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt7, file)
    
    
    
################# Experiment 10: A linear increase in uniform (rate) tax on land value #################

# same initial and terminal condition as before
# save
with open(DYNAMICS_DIR / 'exprmt10_init.json', 'w') as file:
    json.dump(init_ss_exprmt7, file)

# save
with open(DYNAMICS_DIR / 'exprmt10_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt7, file)
    
    
    
    
################# Experiment 11: A sudden jump in tax on land transaction #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt11 = json.load(file)
init_ss_exprmt11 = sol_taxeq0.copy()

# save
with open(DYNAMICS_DIR / 'exprmt11_init.json', 'w') as file:
    json.dump(init_ss_exprmt11, file)


# compute steady state 2
taxes_exprmt11_end = taxes_eq0.copy()
taxes_exprmt11_end['tau_LDV_sale'] = 0.02

terminal_ss_exprmt11 = solve_ss(params_exog, params_endog, taxes_exprmt11_end, init_guess)

# save
with open(DYNAMICS_DIR / 'exprmt11_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt11, file)





################# Experiment 12: A transient (one-period) increase in capital tax to 1.1% #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt12 = json.load(file)
init_ss_exprmt12 = sol_taxeq0.copy()

# save
with open(DYNAMICS_DIR / 'exprmt12_init.json', 'w') as file:
    json.dump(init_ss_exprmt12, file)
# save
with open(DYNAMICS_DIR / 'exprmt12_terminal.json', 'w') as file:
    json.dump(init_ss_exprmt12, file)






################# Experiment 13: A linear increase in tax on raw land surface to 50% #################

# set taxes and/or shock to variables
# with open('sol_uncons_taxeq0.json') as file:
#     init_ss_exprmt13 = json.load(file)
init_ss_exprmt13 = sol_taxeq0.copy()

# save
with open(DYNAMICS_DIR / 'exprmt13_init.json', 'w') as file:
    json.dump(init_ss_exprmt13, file)
    
# compute steady state 2
taxes_exprmt13_end = taxes_eq0.copy()
taxes_exprmt13_end['tau_LR_surface'] = 0.1

terminal_ss_exprmt13 = solve_ss(params_exog, params_endog, taxes_exprmt13_end, init_guess)

# save
with open(DYNAMICS_DIR / 'exprmt13_terminal.json', 'w') as file:
    json.dump(terminal_ss_exprmt13, file)
