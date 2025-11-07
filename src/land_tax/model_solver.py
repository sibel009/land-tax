import numpy as np
from scipy.optimize import fsolve, brentq, least_squares, root, minimize
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import warnings
warnings.filterwarnings("error")


# -------------------------------------------------------
# --------------- specify the environment ---------------
# -------------------------------------------------------

class Environment:
    def __init__(self, params_exog, params_endog, constrained=0):
            
            # parameters that we keep as fixed
            self.aleph =    params_exog['aleph']
            self.n =        params_exog['n']
            self.beta =     params_exog['beta']
            self.omega =    params_exog['omega']
            self.varrho =   params_exog['varrho']    # rentier population weight
            self.gamma_C =  params_exog['gamma_C']
            self.gamma_R =  params_exog['gamma_R']
            self.delta_K =  params_exog['delta_K']
            self.delta_S =  params_exog['delta_S']
            self.alpha_CH = params_exog['alpha_CH']
            self.sigma =    params_exog['sigma']
            self.a_L =      params_exog['a_L']
            self.alpha_K =  params_exog['alpha_K']
            self.A =        params_exog['A']
            self.rho_h =    params_exog['rho_h']
            self.rho_H =    params_exog['rho_H']
            
            # parameters that we want to calibrate on
            self.cost_L1 =   params_endog['cost_L1']
            self.cost_L2 =   params_endog['cost_L2']
            self.Lbar   =   params_endog['Lbar']
            
            if constrained==0:
                self.R_R    =   params_endog['R_R']
            else:
                self.R_R0   =   params_endog['R_R0']
                self.R_R1   =   params_endog['R_R1']
                self.kappa  =   params_endog['kappa']
                self.eta_b  =   params_endog['eta_b']
            

    # ------------ useful functions ----------------
    # utility
    def u(self, c,h):
        alpha_CH, sigma = self.alpha_CH, self.sigma
        
        u = ((c**alpha_CH)*(h**(1-alpha_CH)))**(1-sigma) / (1-sigma) 
        return u
    
    def uR(self, c):
        sigma = self.sigma
        u = c**(1-sigma) / (1-sigma)
        return u
    
    def duR_dc(self, c):
        sigma = self.sigma
        return c**(-sigma)

    # marginal utility wrt c/C
    def du_dc(self, c,h):
        alpha_CH, sigma = self.alpha_CH, self.sigma
        
        duc = (alpha_CH*(c**(alpha_CH-1))*(h**(1-alpha_CH)))*((c**alpha_CH)*h**(1-alpha_CH))**(-sigma)
        return duc

    # marginal utility wrt h/H
    def du_dh(self, c,h):
        alpha_CH, sigma = self.alpha_CH, self.sigma
        
        duh = ((1-alpha_CH)*(c**alpha_CH)*(h**(-alpha_CH)))*(((c**alpha_CH)*(h**(1-alpha_CH)))**(-sigma))
        return duh

    # housing production
    def h_prod(self, l,s):
        a_L = self.a_L
        H = (l**a_L)*(s**(1-a_L))
        return H

    # marginal product wrt l/L
    def dh_prod_dl(self, l,s):
        a_L = self.a_L
        dH_dl = a_L*(l**(a_L-1))*(s**(1-a_L))
        return dH_dl

    # marginal product wrt s/S
    def dh_prod_ds(self, l,s):
        a_L = self.a_L
        dH_ds = (1-a_L)*(l**a_L)*(s**(-a_L))
        return dH_ds
    
    # production function
    def prod(self, k):
        alpha_K = self.alpha_K
        F = self.A*(k)**(alpha_K)
        return F

    # marginal product wrt k
    def dK_prod(self, k):
        alpha_K, A = self.alpha_K, self.A
        dF = (alpha_K)*A*(k)**(alpha_K-1.0)
        return dF

    # # cost of land expansion
    # def cost_landexp(self, x):
    #     """
    #     linear for now
    #     return: cost, marginal cost
    #     """
    #     c_L = self.cost_L
    #     return c_L * x, c_L

    # welfare
    def welfare(self, vars):
        """
        compute welfare: omega*u(c,h) + gamma_C*u(C,H) + varrho*gamma_R*self.uR(C_R)
        self.u(c,h) adjusts h in per worker term
        """
        omega, varrho, gamma_C, gamma_R = self.omega, self.varrho, self.gamma_C, self.gamma_R

        c=vars['c']
        C=vars['C']
        C_R = vars['C_R']
        l=vars['l']
        L=vars['L']
        s=vars['s']
        S=vars['S']
        H=self.h_prod(L,S)
        h=self.h_prod(l,s)

        welfare=omega*self.u(c,h/self.omega)+gamma_C*self.u(C,H)+varrho*gamma_R*self.uR(C_R)
        
        return welfare
        
        

# -------------------------------------------------------
# ---------- calibrate and solve steady state -----------
# -------------------------------------------------------

def solve_ss(params_exog, params_endog, taxes, init_guess, calib=0, targets=None, uniformval=0):
    # If calib == 1, take params_endog as initial guess for param values to calibrate on
    # weight adds more weight to the equilibrium conditions than calib targets
    
    # ---------- Step 1: unpack params and initial guess -----------
    c   = init_guess['c']
    C   = init_guess['C']
    C_R = init_guess['C_R']
    s   = init_guess['s']
    S   = init_guess['S']
    l   = init_guess['l']
    L   = init_guess['L']
    LR  = init_guess['LR']
    
    # ubs = [np.inf,      # c
    #        np.inf,      # C
    #        np.inf,      # C_R
    #        np.inf,      # s
    #        np.inf,      # S
    #        np.inf,      # l
    #        np.inf,      # L
    #        np.inf,      # LR
    #        np.inf,      # cost_L
    #        np.inf,      # R_R
    #        np.inf,      # Lbar
    #        ]
    # lbs = [0,      # c
    #        0,      # C
    #        0,      # C_R
    #        0,      # s
    #        0,      # S
    #        0,      # l
    #        0,      # L
    #        0,      # LR      -np.inf    forbid negative LR
    #        0,      # cost_L
    #        0,      # R_R
    #        0,      # Lbar
    #        ]
    
    # ---------- Step 2: solve for steady state variables ----------
    if calib==1:
        
        def func(x):
            return ss_res(params_exog, None, x, taxes, calib, targets, weight=1e8)[0]
        
        # endogenous params initial guess
        cost_L =   params_endog['cost_L']
        R_R    =   params_endog['R_R']
        Lbar   =   params_endog['Lbar']
        
        x0 = np.array([c, C, C_R, s, S, l, L, LR, cost_L, R_R, Lbar])
        
        # minimize least squares
        result = least_squares(func, x0, bounds=(1e-8, np.inf),
                            #    bounds=(lbs, ubs),
                            #    ftol=1e-15, xtol=1e-15, gtol=1e-15, 
                               verbose=0)
        sol = result.x

        _, output = ss_res(params_exog, None, sol, taxes, calib, targets, weight=1e8)

        # get loss
        Eq_loss = output['Eq_loss']     # equilibrium condition loss
        Tg_loss = output['Tg_loss']     # target loss
        
        if output['Eq_loss'] >= 1e-5:
            print(f'Invalid equilibrium solution: Eq_loss = {Eq_loss}')
            output['sol_flag'] = 1
        else:
            output['sol_flag'] = 0
            
        # print loss of eq conditions and targets
        print('\n----------- Loss -----------')
        print('Equilibrium loss: ', Eq_loss)
        print('Target loss: ', Tg_loss)
        
        # unpack solution - endog params
        params_calibrated = {
            'cost_L' : sol[8],
            'R_R':  sol[9],
            'Lbar': sol[10]
        }
        
        return output, params_calibrated

    elif uniformval==1:
        def func(x):
            return ss_res_uniformval(params_exog, params_endog, x, taxes, calib=0, targets=None, weight=1)[0]

        tau_LR_value = 0.5
        x0 = np.array([c, C, C_R, s, S, l, L, LR, tau_LR_value])
    
        # minimize least squares
        result = least_squares(func, x0, bounds=(1e-8, np.inf), 
                            #    bounds=(lbs[:8], ubs[:8]),
                            #    ftol=1e-15, xtol=1e-15, gtol=1e-15, 
                               verbose=0)
        sol = result.x
        
        # func(result.x)
        
        tau_LR_value = sol[8]
        _, output = ss_res_uniformval(params_exog, params_endog, sol, taxes, calib=0, targets=None, weight=1)
        
        output['tau_LR_value'] = tau_LR_value
        # get loss
        Eq_loss = output['Eq_loss']     # equilibrium condition loss
        Tg_loss = output['Tg_loss']     # target loss
        
        if output['Eq_loss'] >= 1e-5:
            # print(f'Invalid equilibrium solution: Eq_loss = {Eq_loss}')
            output['sol_flag'] = 1
        elif output['p_LR'] <=  1e-7:       # negative price
            output['sol_flag'] = 1
        else:
            output['sol_flag'] = 0
            
        return output
        
    else:
        
        def func(x):
            return ss_res(params_exog, params_endog, x, taxes, calib=0, targets=None, weight=1)[0]

        
        x0 = np.array([c, C, C_R, s, S, l, L, LR])
    
        # minimize least squares
        result = least_squares(func, x0, bounds=(1e-8, np.inf), 
                            #    bounds=(lbs[:8], ubs[:8]),
                            #    ftol=1e-15, xtol=1e-15, gtol=1e-15, 
                               verbose=0)
        sol = result.x
        
        _, output = ss_res(params_exog, params_endog, sol, taxes, calib=0, targets=None, weight=1)
        
        # get loss
        Eq_loss = output['Eq_loss']     # equilibrium condition loss
        Tg_loss = output['Tg_loss']     # target loss
        
        if output['Eq_loss'] >= 1e-5:
            # print(f'Invalid equilibrium solution: Eq_loss = {Eq_loss}')
            output['sol_flag'] = 1
        elif output['p_LR'] <=  1e-7:       # negative price
            output['sol_flag'] = 1
        else:
            output['sol_flag'] = 0
            
        return output


 
def ss_res(params_exog, params_endog, x, taxes, calib, targets, weight):
    """
    A system of steady state equations for decentralized program
    ---------
    Inputs
        vars : a vector of endogenous vars in SP program
            [c, C, s, S, l, L, p_LDV]
    Return
        res : a loss vector of length 
    """
    
    # unpack the endogenous variables
    c,C,C_R,s,S,l,L,LR  = x[:8]
    
        
    if calib==1:
        # unpack the endogenous parameters
        params_endog = {
            'cost_L': x[8],
            'R_R':    x[9],
            'Lbar':   x[10],
        }
    
    # create environment
    env = Environment(params_exog, params_endog)

    # unpack all parameters
    omega, n, aleph, beta = env.omega, env.n, env.aleph, env.beta
    delta_K, delta_S = env.delta_K, env.delta_S
    rho_H, rho_h = env.rho_H, env.rho_h
    alpha_K = env.alpha_K
    varrho = env.varrho
    cost_L1 = env.cost_L1
    cost_L2 = env.cost_L2
    gamma_C, gamma_R = env.gamma_C, env.gamma_R
    
    # total land
    Lbar = env.Lbar
    R_R = env.R_R

    # taxes
    tau_s_inv        = taxes['tau_s_inv']
    tau_S_inv        = taxes['tau_S_inv']  
    tau_K_inv        = taxes['tau_K_inv']
    tau_K            = taxes['tau_K']
    tau_l_surface    = taxes['tau_l_surface']
    tau_l_value      = taxes['tau_l_value']
    tau_L_surface    = taxes['tau_L_surface']
    tau_L_value      = taxes['tau_L_value']
    tau_LR_surface   = taxes['tau_LR_surface']
    tau_LR_value     = taxes['tau_LR_value']
    tau_LDV_sale     = taxes['tau_LDV_sale']
    tau_LR_sale      = taxes['tau_LR_sale']
    tau_H            = taxes['tau_H']
    tau_HI           = taxes['tau_HI']
    tau_c            = taxes.get('tau_c', 0.0)
    tau_ch           = taxes.get('tau_ch', tau_c)
    tau_C            = taxes.get('tau_C', 0.0)
    tau_CH           = taxes.get('tau_CH', 0.0)
    tau_CR           = taxes.get('tau_CR', 0.0)
    tau_D_K          = taxes['tau_D_K']
    tau_D_S          = taxes['tau_D_S']
    tau_D_s          = taxes['tau_D_s']  
    tau_D_H          = taxes['tau_D_H']
    tau_D_h          = taxes['tau_D_h']
    tau_D_L          = taxes['tau_D_L'] 
    tau_D_l          = taxes['tau_D_l'] 
    tau_D_LR         = taxes['tau_D_LR']
    tau_inher_K      = taxes['tau_inher_K']
    tau_inher_S      = taxes['tau_inher_S']
    tau_inher_s      = taxes['tau_inher_s']
    tau_inher_H      = taxes['tau_inher_H']
    tau_inher_h      = taxes['tau_inher_h']
    tau_inher_L      = taxes['tau_inher_L']
    tau_inher_l      = taxes['tau_inher_l']
    tau_inher_LR     = taxes['tau_inher_LR']

    # dilution
    d_K     = n + aleph*tau_inher_K
    d_H     = n + aleph*tau_inher_H
    d_h     = n + aleph*tau_inher_h
    d_S     = n + aleph*tau_inher_S
    d_s     = n + aleph*tau_inher_s
    d_L     = n + aleph*tau_inher_L
    d_l     = n + aleph*tau_inher_l
    d_LR    = n + aleph*tau_inher_LR
    
    
    # solve steady state capital level   
    def res_K(x): 
        return (
        (1+tau_K_inv*(1+n)+d_K*(1+tau_D_K)) 
        - beta*(tau_K_inv*(1-delta_K) + (1-tau_K)*(1-delta_K+env.dK_prod(x)))
        )
    
    k = brentq(res_K, 1e-5, 100)
    
    # per capita capital stock
    K = k*(1+omega)           # *(1+omega)

    # ----- intermediate values -----
    H = env.h_prod(L,S)
    h = env.h_prod(l,s)
    LDV = l + L


    # marginal derivatives
    du_dc = env.du_dc(c,h/omega)
    du_dh = env.du_dh(c,h/omega)
    dU_dC = env.du_dc(C,H)
    dU_dH = env.du_dh(C,H)
    duR_dC_R = env.duR_dc(C_R)
    dH_dL = env.dh_prod_dl(L,S)
    dH_dS = env.dh_prod_ds(L,S)
    dh_dl = env.dh_prod_dl(l,s)
    dh_ds = env.dh_prod_ds(l,s)


    # prices
    R_H_gross = du_dh/du_dc     # using worker FOC
    R_H_net = (1-tau_H)*R_H_gross
    R_K_gross = env.dK_prod(k) + 1 - delta_K
    R_K_net = (1-tau_K)*R_K_gross
    wage = (1-alpha_K)*env.prod(k)
    Y = (1+omega)*env.prod(k)    # total output         (1+omega)

    p_LR = (
        (R_R*beta-tau_LR_surface)/
        ((1-tau_LR_sale)*(1+n-beta)+tau_LR_value+aleph*tau_inher_LR+d_LR*tau_D_LR)
    )

    # shadow price
    p_LDV = (1+tau_LDV_sale)*p_LR + cost_L1 + cost_L2*n*LDV
    
    # if uniformval==1:
    #     # compute implied tax rate on L, l value
    #     tau_L_value = tau_LR_value * p_LR / p_LDV
    #     tau_l_value = tau_LR_value * p_LR / p_LDV


    # multipliers
    LM_C = dU_dC
    LM_L = LM_C * (1+n-beta) * ((1+tau_LDV_sale)* p_LR + cost_L1 + cost_L2*n*LDV)

    LM_R_C = duR_dC_R
    LM_R_L = (1-tau_LR_sale) * (1+n-beta) * p_LR * LM_R_C
    

    # ----- government revenues -----
    # tax
    T_Ss_inv = tau_S_inv*(n+delta_S)*S + tau_s_inv*(n+delta_S)*s
    T_K_inv = tau_K_inv*(n+delta_K)*K
    T_K     = tau_K*R_K_gross*K
    T_L = (
          (tau_L_surface+tau_L_value*p_LDV)*L
        + (tau_l_surface+tau_l_value*p_LDV)*l
        +  tau_LDV_sale*p_LR*n*LDV
        )
    T_L_R = varrho*(tau_LR_surface+tau_LR_value*p_LR)*LR + tau_LR_sale*p_LR*n*LDV
    T_H = tau_H*R_H_gross*h + tau_HI*R_H_gross*H
    T_C = (
        varrho * tau_CR * C_R
        + tau_C * C
        + omega * tau_c * c
        + omega * tau_ch * R_H_gross * h
        + tau_CH * R_H_gross * H
    )

    # inheritance tax
    T_inher = (
        aleph*tau_inher_K*K + 
        aleph*tau_inher_S*S + 
        aleph*tau_inher_s*s + 
        aleph*tau_inher_H*H*R_H_gross/rho_H +
        aleph*tau_inher_h*h*R_H_net/rho_h +
        aleph*tau_inher_L*p_LDV*L + 
        aleph*tau_inher_l*p_LDV*l
        )
    T_inher_R = varrho*aleph*tau_inher_LR*p_LR*LR

    # donation tax
    T_D = (
        d_K*tau_D_K*K + 
        d_S*tau_D_S*S + 
        d_s*tau_D_s*s + 
        d_H*tau_D_H*H*R_H_gross/rho_H +
        d_h*tau_D_h*h*R_H_net/rho_h +
        d_L*tau_D_L*p_LDV*L + 
        d_l*tau_D_l*p_LDV*l
        )
    T_D_R = varrho*d_LR*tau_D_LR*p_LR*LR

    # total revenue
    T_total_R = T_L_R + T_inher_R + T_D_R   # in capitalist unit
    T_total_C = T_Ss_inv + T_K_inv + T_K + T_L + T_H + T_inher + T_D

    T_total = T_total_C + T_total_R + T_C

    workers_cons_tax = omega * tau_c * c + omega * tau_ch * R_H_gross * h
    R_net_to_workers_total = T_total - workers_cons_tax
    R_net_to_workers = R_net_to_workers_total / (1 + omega)
    
    
    # national income and GDP
    GDP = Y + R_H_gross*(H+h) + varrho*R_R*LR
    NatIncome_prod = GDP - delta_K*K - delta_S*(S+s)    # GDP net of depreciation
    NatIncome_income = (
        (1+omega)*wage + varrho*R_R*LR + K*(R_K_gross-1) + R_H_net*(H+h)            # (1+omega)
        + T_total - (T_K_inv + T_K) - delta_S*(S+s))
    NatIncome_demand = (
        varrho*C_R + C + omega*c + R_H_gross*H + R_H_net*h + n*K + n*(S+s) 
        + cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2 + R_H_gross*(H+h) + varrho*R_R*LR)

    # some marginal benefit and cost expressions
    marg_benef_H = (
        beta*(
            dU_dH
            - LM_C*
            (
                (tau_CH + tau_HI)
                + aleph*tau_inher_H/rho_H
                + d_H*tau_D_H/rho_H
            )
            * R_H_gross
        )
    )
    marg_benef_h = (
        beta*LM_C*(1-tau_H)*(1 - aleph*tau_inher_h/rho_h - d_h*tau_D_h/rho_h) * R_H_gross
    )
    tax_wedge_L  = tau_L_surface + (tau_L_value + d_L*tau_D_L + aleph*tau_inher_L)*p_LDV
    tax_wedge_l  = tau_l_surface + (tau_l_value + d_l*tau_D_l + aleph*tau_inher_l)*p_LDV
    
    tax_wedge_S = 1 + tau_S_inv*(1+n) + d_S*(1+tau_D_S)
    tax_wedge_s = 1 + tau_s_inv*(1+n) + d_s*(1+tau_D_s)

    # loss vector
    res = np.zeros(8)
    
    # -------------------------------------------------------------
    # ------------------ steady state conditions ------------------
    # -------------------------------------------------------------

    # ------ capitalist's FOC ------
    # Euler on L
    res[0] = marg_benef_H*dH_dL - LM_C*tax_wedge_L - LM_L

    # Euler on l
    res[1] = marg_benef_h*dh_dl - LM_C*tax_wedge_l - LM_L

    # Euler on S
    res[2] = LM_C*tax_wedge_S - beta*LM_C*(1-delta_S)*(1+tau_S_inv) - marg_benef_H*dH_dS
    
    # Euler on s
    res[3] = LM_C*tax_wedge_s - beta*LM_C*(1-delta_S)*(1+tau_s_inv) - marg_benef_h*dh_ds

    # BCs
    res[4] = (Y - omega*wage + R_H_gross*h
                - (n+delta_K)*K - (n+delta_S)*(S+s) - C
                - n*LDV*p_LR - cost_L1*n*LDV - 1/2*cost_L2*(n*LDV)**2 - T_total_C + R_net_to_workers)    #  + T_total/(1+omega)

    res[5] = (R_R*LR + n*LDV*p_LR/varrho
                - C_R - T_total_R/varrho)

    res[6] = LDV + varrho*LR - Lbar

    # worker
    worker_outlays = (1 + tau_c) * c + (1 + tau_ch) * (R_H_gross/omega) * h
    res[7] = worker_outlays - wage - R_net_to_workers       # - T_total/(1+omega)

    res_worker_intraperiod = (
        (du_dh / (du_dc * R_H_gross)) - ((1 + tau_ch) / (1 + tau_c))
    )

    tau_CH_implied = (
        (tau_C / (1 + tau_C)) * (dU_dH / (dU_dC * R_H_gross))
        if (1 + tau_C) != 0
        else 0.0
    )
    res_tauCH_gap = tau_CH - tau_CH_implied

    
    # add weight to these conditions
    res[:8] = res[:8] * weight
    
    # -------------------------------------------------------------
    # ------------------- calibration targets ---------------------
    # -------------------------------------------------------------
    LR_GDP_ratio    = (p_LR * varrho * LR / GDP)
    dev_GDP_ratio   = ((p_LDV*(L + l)) / GDP)
    real_estate_GDP_ratio = (R_H_gross/(1-beta) * (H + h) / GDP)
    K_Y_ratio             = (K/Y)
    pLDV_pLR_ratio    = (p_LDV/p_LR)
    LDV_LR_ratio        = ((L+l)/(varrho*LR))
    

    if calib==1:
        # raw land value/GDP
        res[8] = np.log(LR_GDP_ratio / targets['LR_GDP_ratio'])
        
        # dev land value/GDP
        res[9] = np.log(dev_GDP_ratio / targets['dev_GDP_ratio'])
        
        # real estate/GDP
        res[10] = np.log(real_estate_GDP_ratio / targets['real_estate_GDP_ratio'])
        
        # capital/output
        res[11] = np.log(K_Y_ratio / targets['K_Y_ratio'])
        
        # # land price ratio  --------?
        # res[12] = np.log(p_LR_p_LDV_ratio / targets['pLDV_pLR_ratio'])
        
        # # land size ratio  --------?
        # res[13] = np.log(LDV_LR_ratio / targets['LDV_LR_ratio'])
    
    
    # -------------------------------------------------------------
    # ---------------------- output to keep -----------------------
    # -------------------------------------------------------------
    output = {
        # solutions
        'c'  : c,
        'C'  : C, 
        'C_R': C_R,
        's'  : s,
        'S'  : S,
        'l'  : l,
        'L'  : L,
        'LR' : LR,
        
        # prices
        'R_H_gross' :   R_H_gross,
        'R_H_net' :     R_H_net,
        'R_K_gross' :   R_K_gross,
        'R_K_net' :     R_K_net,
        'p_LDV'   :   p_LDV,
        'p_LR'  :   p_LR,
        'wage' :        wage,
        
        # agg variables
        'K' : K,
        'H' : H,
        'h' : h,
        'Y' : Y,
        'LDV'     : LDV,
        'LDV_exp' : LDV*n,
        'GDP' : GDP,
        'NatIncome_prod':   NatIncome_prod,
        'NatIncome_income':  NatIncome_income,
        'NatIncome_demand':  NatIncome_demand,
        'marginal_cost': cost_L1 + cost_L2*n*LDV,
        'total_cost': cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2,
        
        # multipliers
        'LM_C' : LM_C,
        'LM_L' : LM_L,
        'LM_R_C':    LM_R_C,
        'LM_R_L':    LM_R_L,
        
        # taxes
        'T_Ss_inv' : T_Ss_inv,
        'T_K_inv'  : T_K_inv,
        'T_K'      : T_K,
        'T_L'      : T_L,
        'T_L_R'    : T_L_R,
        'T_H'      : T_H,
        'T_C'      : T_C,
        'T_inher'  : T_inher,
        'T_inher_R': T_inher_R,
        'T_D'      : T_D,
        'T_D_R'    : T_D_R,
        'T_total_C' : T_total_C,
        'T_total_R' : T_total_R,
        'T_total'   : T_total,
        'workers_cons_tax': workers_cons_tax,
        'R_net_to_workers_total': R_net_to_workers_total,
        'R_net_to_workers': R_net_to_workers,
        
        # welfare 
        'util_c'   : env.u(c, h/omega),
        'util_C'   : env.u(C, H),
        'util_CR'  : env.uR(C_R),
        'welfare'  : omega*env.u(c,h/env.omega) 
            + gamma_C*env.u(C,H)+varrho*gamma_R*env.uR(C_R),

        # derivatives
        'du_dc'    : du_dc,
        'du_dh'    : du_dh,
        'dU_dC'    : dU_dC,
        'dU_dH'    : dU_dH,
        'duR_dC_R' : duR_dC_R,
        'dH_dL': dH_dL,
        'dH_dS': dH_dS,
        'dh_dl': dh_dl,
        'dh_ds': dh_ds,
        
        # targets
        'LR_GDP_ratio': LR_GDP_ratio,
        'dev_GDP_ratio': dev_GDP_ratio,
        'real_estate_GDP_ratio': real_estate_GDP_ratio,
        'K_Y_ratio': K_Y_ratio,
        'pLDV_pLR_ratio': pLDV_pLR_ratio,
        'LDV_LR_ratio': LDV_LR_ratio,
        'worker_outlays': worker_outlays,
        'res_worker_intraperiod': res_worker_intraperiod,
        'tau_CH_implied': tau_CH_implied,
        'res_tauCH_gap': res_tauCH_gap,

        # loss
        'Eq_loss': np.sqrt(np.sum(np.square(res[:8]/weight))),
        'Tg_loss': np.sqrt(np.sum(np.square(res[8:])))
    }
    
    # also append the tax values to the output
    output.update(taxes)
    output.update(params_exog)
    output.update(params_endog)
    
    return res, output





def ss_res_uniformval(params_exog, params_endog, x, taxes, calib, targets, weight):
    """
    A system of steady state equations for decentralized program
    ---------
    Inputs
        vars : a vector of endogenous vars in SP program
            [c, C, s, S, l, L, p_LDV]
    Return
        res : a loss vector of length 
    """
    
    # unpack the endogenous variables
    c,C,C_R,s,S,l,L,LR,tau_LR_value  = x[:9]
    
    # create environment
    env = Environment(params_exog, params_endog)

    # unpack all parameters
    omega, n, aleph, beta = env.omega, env.n, env.aleph, env.beta
    delta_K, delta_S = env.delta_K, env.delta_S
    rho_H, rho_h = env.rho_H, env.rho_h
    alpha_K = env.alpha_K
    varrho = env.varrho
    cost_L1, cost_L2 = env.cost_L1, env.cost_L2
    gamma_C, gamma_R = env.gamma_C, env.gamma_R
    
    # total land
    Lbar = env.Lbar
    R_R = env.R_R

    # taxes
    tau_s_inv        = taxes['tau_s_inv']
    tau_S_inv        = taxes['tau_S_inv']  
    tau_K_inv        = taxes['tau_K_inv']
    tau_K            = taxes['tau_K']
    tau_l_surface    = taxes['tau_l_surface']
    tau_l_value      = taxes['tau_l_value']
    tau_L_surface    = taxes['tau_L_surface']
    tau_L_value      = taxes['tau_L_value']
    tau_LR_surface   = taxes['tau_LR_surface']
    tau_LDV_sale     = taxes['tau_LDV_sale']
    tau_LR_sale      = taxes['tau_LR_sale']
    tau_H            = taxes['tau_H']
    tau_HI           = taxes['tau_HI']
    tau_D_K          = taxes['tau_D_K']
    tau_D_S          = taxes['tau_D_S']
    tau_D_s          = taxes['tau_D_s']  
    tau_D_H          = taxes['tau_D_H']
    tau_D_h          = taxes['tau_D_h']
    tau_D_L          = taxes['tau_D_L'] 
    tau_D_l          = taxes['tau_D_l'] 
    tau_D_LR         = taxes['tau_D_LR']
    tau_inher_K      = taxes['tau_inher_K']
    tau_inher_S      = taxes['tau_inher_S']
    tau_inher_s      = taxes['tau_inher_s']
    tau_inher_H      = taxes['tau_inher_H']
    tau_inher_h      = taxes['tau_inher_h']
    tau_inher_L      = taxes['tau_inher_L']
    tau_inher_l      = taxes['tau_inher_l']
    tau_inher_LR     = taxes['tau_inher_LR']

    # dilution
    d_K     = n + aleph*tau_inher_K
    d_H     = n + aleph*tau_inher_H
    d_h     = n + aleph*tau_inher_h
    d_S     = n + aleph*tau_inher_S
    d_s     = n + aleph*tau_inher_s
    d_L     = n + aleph*tau_inher_L
    d_l     = n + aleph*tau_inher_l
    d_LR    = n + aleph*tau_inher_LR
    
    
    # solve steady state capital level   
    def res_K(x): 
        return (
        (1+tau_K_inv*(1+n)+d_K*(1+tau_D_K)) 
        - beta*(tau_K_inv*(1-delta_K) + (1-tau_K)*(1-delta_K+env.dK_prod(x)))
        )
    
    k = brentq(res_K, 1e-5, 100)
    
    # per capita capital stock
    K = k*(1+omega)           # *(1+omega)

    # ----- intermediate values -----
    H = env.h_prod(L,S)
    h = env.h_prod(l,s)
    LDV = l + L


    # marginal derivatives
    du_dc = env.du_dc(c,h/omega)
    du_dh = env.du_dh(c,h/omega)
    dU_dC = env.du_dc(C,H)
    dU_dH = env.du_dh(C,H)
    duR_dC_R = env.duR_dc(C_R)
    dH_dL = env.dh_prod_dl(L,S)
    dH_dS = env.dh_prod_ds(L,S)
    dh_dl = env.dh_prod_dl(l,s)
    dh_ds = env.dh_prod_ds(l,s)


    # prices
    R_H_gross = du_dh/du_dc     # using worker FOC
    R_H_net = (1-tau_H)*R_H_gross
    R_K_gross = env.dK_prod(k) + 1 - delta_K
    R_K_net = (1-tau_K)*R_K_gross
    wage = (1-alpha_K)*env.prod(k)
    Y = (1+omega)*env.prod(k)    # total output         (1+omega)

    p_LR = (
        (R_R*beta-tau_LR_surface)/
        ((1-tau_LR_sale)*(1+n-beta)+tau_LR_value+aleph*tau_inher_LR+d_LR*tau_D_LR)
    )

    # shadow price
    p_LDV = (1+tau_LDV_sale)*p_LR + cost_L1 + cost_L2*n*LDV


    # multipliers
    LM_C = dU_dC
    LM_L = LM_C * (1+n-beta) * ((1+tau_LDV_sale)* p_LR + cost_L1 + cost_L2*n*LDV)

    LM_R_C = duR_dC_R
    LM_R_L = (1-tau_LR_sale) * (1+n-beta) * p_LR * LM_R_C
    

    # ----- government revenues -----
    # tax
    T_Ss_inv = tau_S_inv*(n+delta_S)*S + tau_s_inv*(n+delta_S)*s
    T_K_inv = tau_K_inv*(n+delta_K)*K
    T_K     = tau_K*R_K_gross*K
    T_L = (
          (tau_L_surface+tau_L_value*p_LDV)*L 
        + (tau_l_surface+tau_l_value*p_LDV)*l
        +  tau_LDV_sale*p_LR*n*LDV
        )
    T_L_R = varrho*(tau_LR_surface+tau_LR_value*p_LR)*LR + tau_LR_sale*p_LR*n*LDV
    T_H = tau_H*R_H_gross*h + tau_HI*R_H_gross*H

    # inheritance tax
    T_inher = (
        aleph*tau_inher_K*K + 
        aleph*tau_inher_S*S + 
        aleph*tau_inher_s*s + 
        aleph*tau_inher_H*H*R_H_gross/rho_H +
        aleph*tau_inher_h*h*R_H_net/rho_h +
        aleph*tau_inher_L*p_LDV*L + 
        aleph*tau_inher_l*p_LDV*l
        )
    T_inher_R = varrho*aleph*tau_inher_LR*p_LR*LR

    # donation tax
    T_D = (
        d_K*tau_D_K*K + 
        d_S*tau_D_S*S + 
        d_s*tau_D_s*s + 
        d_H*tau_D_H*H*R_H_gross/rho_H +
        d_h*tau_D_h*h*R_H_net/rho_h +
        d_L*tau_D_L*p_LDV*L + 
        d_l*tau_D_l*p_LDV*l
        )
    T_D_R = varrho*d_LR*tau_D_LR*p_LR*LR

    # total revenue
    T_total_R = T_L_R + T_inher_R + T_D_R   # in capitalist unit
    T_total_C = T_Ss_inv + T_K_inv + T_K + T_L + T_H + T_inher + T_D

    T_total = T_total_C + T_total_R + T_C

    workers_cons_tax = omega * tau_c * c + omega * tau_ch * R_H_gross * h
    R_net_to_workers_total = T_total - workers_cons_tax
    R_net_to_workers = R_net_to_workers_total / (1 + omega)
    
    
    # national income and GDP
    GDP = Y + R_H_gross*(H+h) + varrho*R_R*LR
    NatIncome_prod = GDP - delta_K*K - delta_S*(S+s)    # GDP net of depreciation
    NatIncome_income = (
        (1+omega)*wage + varrho*R_R*LR + K*(R_K_gross-1) + R_H_net*(H+h)            # (1+omega)
        + T_total - (T_K_inv + T_K) - delta_S*(S+s))
    NatIncome_demand = (
        varrho*C_R + C + omega*c + R_H_gross*H + R_H_net*h + n*K + n*(S+s) 
        + cost_L1*n*LDV+ 1/2*cost_L2*(n*LDV)**2 + R_H_gross*(H+h) + varrho*R_R*LR)

    # some marginal benefit and cost expressions
    marg_benef_H = (
        beta*(
            dU_dH
            - LM_C*
            (
                (tau_CH + tau_HI)
                + aleph*tau_inher_H/rho_H
                + d_H*tau_D_H/rho_H
            )
            * R_H_gross
        )
    )
    marg_benef_h = (
        beta*LM_C*(1-tau_H)*(1 - aleph*tau_inher_h/rho_h - d_h*tau_D_h/rho_h) * R_H_gross
    )
    tax_wedge_L  = tau_L_surface + (tau_L_value + d_L*tau_D_L + aleph*tau_inher_L)*p_LDV
    tax_wedge_l  = tau_l_surface + (tau_l_value + d_l*tau_D_l + aleph*tau_inher_l)*p_LDV
    
    tax_wedge_S = 1 + tau_S_inv*(1+n) + d_S*(1+tau_D_S)
    tax_wedge_s = 1 + tau_s_inv*(1+n) + d_s*(1+tau_D_s)

    # loss vector
    res = np.zeros(9)
    
    # -------------------------------------------------------------
    # ------------------ steady state conditions ------------------
    # -------------------------------------------------------------

    # ------ capitalist's FOC ------
    # Euler on L
    res[0] = marg_benef_H*dH_dL - LM_C*tax_wedge_L - LM_L

    # Euler on l
    res[1] = marg_benef_h*dh_dl - LM_C*tax_wedge_l - LM_L

    # Euler on S
    res[2] = LM_C*tax_wedge_S - beta*LM_C*(1-delta_S)*(1+tau_S_inv) - marg_benef_H*dH_dS
    
    # Euler on s
    res[3] = LM_C*tax_wedge_s - beta*LM_C*(1-delta_S)*(1+tau_s_inv) - marg_benef_h*dh_ds

    # BCs
    res[4] = (Y - omega*wage + R_H_gross*h
                - (n+delta_K)*K - (n+delta_S)*(S+s) - C
                - n*LDV*p_LR - cost_L1*n*LDV - 1/2*(n*LDV)**2*cost_L2 - T_total_C + R_net_to_workers)    #  + T_total/(1+omega)

    res[5] = (R_R*LR + n*LDV*p_LR/varrho
                - C_R - T_total_R/varrho)

    res[6] = LDV + varrho*LR - Lbar

    # worker
    worker_outlays = (1 + tau_c) * c + (1 + tau_ch) * (R_H_gross/omega) * h
    res[7] = worker_outlays - wage - R_net_to_workers       # - T_total/(1+omega)

    res_worker_intraperiod = (
        (du_dh / (du_dc * R_H_gross)) - ((1 + tau_ch) / (1 + tau_c))
    )

    tau_CH_implied = (
        (tau_C / (1 + tau_C)) * (dU_dH / (dU_dC * R_H_gross))
        if (1 + tau_C) != 0
        else 0.0
    )
    res_tauCH_gap = tau_CH - tau_CH_implied


    # add weight to these conditions
    res[:8] = res[:8] * weight
    
    res[8] = p_LDV * tau_L_value - tau_LR_value * p_LR
    
    # -------------------------------------------------------------
    # ------------------- calibration targets ---------------------
    # -------------------------------------------------------------
    LR_GDP_ratio    = (p_LR * varrho * LR / GDP)
    dev_GDP_ratio   = ((p_LDV*(L + l)) / GDP)
    real_estate_GDP_ratio = (R_H_gross/(1-beta) * (H + h) / GDP)
    K_Y_ratio             = (K/Y)
    pLDV_pLR_ratio    = (p_LDV/p_LR)
    LDV_LR_ratio        = ((L+l)/(varrho*LR))
    
    
    # -------------------------------------------------------------
    # ---------------------- output to keep -----------------------
    # -------------------------------------------------------------
    output = {
        # solutions
        'c'  : c,
        'C'  : C, 
        'C_R': C_R,
        's'  : s,
        'S'  : S,
        'l'  : l,
        'L'  : L,
        'LR' : LR,
        
        # prices
        'R_H_gross' :   R_H_gross,
        'R_H_net' :     R_H_net,
        'R_K_gross' :   R_K_gross,
        'R_K_net' :     R_K_net,
        'p_LDV'   :   p_LDV,
        'p_LR'  :   p_LR,
        'wage' :        wage,
        
        # agg variables
        'K' : K,
        'H' : H,
        'h' : h,
        'Y' : Y,
        'LDV'     : LDV,
        'LDV_exp' : LDV*n,
        'GDP' : GDP,
        'NatIncome_prod':   NatIncome_prod,
        'NatIncome_income':  NatIncome_income,
        'NatIncome_demand':  NatIncome_demand,
        'marginal_cost': cost_L1 + cost_L2*n*LDV,
        'total_cost': cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2,
        
        # multipliers
        'LM_C' : LM_C,
        'LM_L' : LM_L,
        'LM_R_C':    LM_R_C,
        'LM_R_L':    LM_R_L,
        
        # taxes
        'T_Ss_inv' : T_Ss_inv,
        'T_K_inv'  : T_K_inv,
        'T_K'      : T_K,
        'T_L'      : T_L,
        'T_L_R'    : T_L_R,
        'T_H'      : T_H,
        'T_C'      : T_C,
        'T_inher'  : T_inher,
        'T_inher_R': T_inher_R,
        'T_D'      : T_D,
        'T_D_R'    : T_D_R,
        'T_total_C' : T_total_C,
        'T_total_R' : T_total_R,
        'T_total'   : T_total,
        'workers_cons_tax': workers_cons_tax,
        'R_net_to_workers_total': R_net_to_workers_total,
        'R_net_to_workers': R_net_to_workers,
        
        # welfare 
        'util_c'   : env.u(c, h/omega),
        'util_C'   : env.u(C, H),
        'util_CR'  : env.uR(C_R),
        'welfare'  : omega*env.u(c,h/env.omega) 
            + gamma_C*env.u(C,H)+varrho*gamma_R*env.uR(C_R),

        # derivatives
        'du_dc'    : du_dc,
        'du_dh'    : du_dh,
        'dU_dC'    : dU_dC,
        'dU_dH'    : dU_dH,
        'duR_dC_R' : duR_dC_R,
        'dH_dL': dH_dL,
        'dH_dS': dH_dS,
        'dh_dl': dh_dl,
        'dh_ds': dh_ds,
        
        # targets
        'LR_GDP_ratio': LR_GDP_ratio,
        'dev_GDP_ratio': dev_GDP_ratio,
        'real_estate_GDP_ratio': real_estate_GDP_ratio,
        'K_Y_ratio': K_Y_ratio,
        'pLDV_pLR_ratio': pLDV_pLR_ratio,
        'LDV_LR_ratio': LDV_LR_ratio,
        'worker_outlays': worker_outlays,
        'res_worker_intraperiod': res_worker_intraperiod,
        'tau_CH_implied': tau_CH_implied,
        'res_tauCH_gap': res_tauCH_gap,
        
        # loss
        'Eq_loss': np.sqrt(np.sum(np.square(res[:8]/weight))),
        'Tg_loss': np.sqrt(np.sum(np.square(res[8:])))
    }
    
    # also append the tax values to the output
    output.update(taxes)
    output.update(params_exog)
    output.update(params_endog)
    output['tau_LR_value'] = tau_LR_value
    
    return res, output









####################################################################################
###################### Constrained construtible land case ##########################
####################################################################################


def solve_ss_constructible(params_exog, params_endog, taxes, init_guess):
    # If calib == 1, take params_endog as initial guess for param values to calibrate on
    # weight adds more weight to the equilibrium conditions than calib targets
    
    # ---------- Step 1: unpack params and initial guess -----------
    c   = init_guess['c']
    C   = init_guess['C']
    C_R = init_guess['C_R']
    s   = init_guess['s']
    S   = init_guess['S']
    l   = init_guess['l']
    L   = init_guess['L']
    LR_1  = init_guess['LR_1']
    Psi_R  = init_guess['Psi_R']
    Psi_C  = init_guess['Psi_C']
    
    # ---------- Step 2: solve for steady state variables ----------
    def func(x):
        return ss_res_nonbinding(params_exog, params_endog, x, taxes)[0]
    
    x0 = np.array([c, C, C_R, s, S, l, L, LR_1])

    # print('======= solve nonbinding case ======')
    # minimize least squares
    result = least_squares(func, x0, 
                           bounds=(1e-8, np.inf), 
                        #    bounds=(lbs[:8], ubs[:8]),
                        #    ftol=1e-15, xtol=1e-15, gtol=1e-15, 
                            verbose=0)
    sol = result.x
    
    _, output = ss_res_nonbinding(params_exog, params_endog, sol, taxes)
    
    loss = output['Eq_loss']
    # print(f'Loss: {loss}')
    
    # check leftover
    leftover = output['leftover']
    
    if leftover >=0:
        print(f'We are good. Leftover LR_1 is {leftover}.')
        output['Psi_C']    = 0    # assign multiplier
        output['Psi_R']    = 0
        output['sol_flag'] = 2    # there is leftover
    else:
        # print('Constraint not satisfied. More new LDV than LR_1.')
        # print('======= solve binding case ======')

        def func(x):
            return ss_res_binding(params_exog, params_endog, x, taxes)[0]
        
        x0 = np.array([c, C, C_R, s, S, l, Psi_R, Psi_C])
        
        result = least_squares(func, x0, bounds=(1e-8, np.inf), verbose=0)
        sol = result.x 
        
        _, output = ss_res_binding(params_exog, params_endog, sol, taxes)
        
        loss = output['Eq_loss']
        # print(f'Loss: {loss}')
        
        if loss >= 1e-5:
            # print(f'Invalid equilibrium solution: Eq_loss = {Eq_loss}')
            output['sol_flag'] = 1
        elif output['p_LR'] <=  1e-7:       # negative price
            output['sol_flag'] = 1
        else:
            output['sol_flag'] = 0
        
    return output


 
def ss_res_nonbinding(params_exog, params_endog, x, taxes):
    
    # unpack the endogenous variables
    c,C,C_R,s,S,l,L,LR_1  = x[:8]
    
    # create environment
    env = Environment(params_exog, params_endog, constrained=1)

    # unpack all parameters
    omega, n, aleph, beta = env.omega, env.n, env.aleph, env.beta
    delta_K, delta_S = env.delta_K, env.delta_S
    rho_H, rho_h = env.rho_H, env.rho_h
    alpha_K = env.alpha_K
    varrho = env.varrho
    cost_L1, cost_L2 = env.cost_L1, env.cost_L2
    gamma_C, gamma_R = env.gamma_C, env.gamma_R
    
    kappa = env.kappa   # transformation rate
    R_R0, R_R1 = env.R_R0, env.R_R1
    
    # total land
    Lbar = env.Lbar
    LR_0 = (n/(n+kappa) * Lbar) / varrho

    # taxes
    tau_s_inv        = taxes['tau_s_inv']
    tau_S_inv        = taxes['tau_S_inv']  
    tau_K_inv        = taxes['tau_K_inv']
    tau_K            = taxes['tau_K']
    tau_l_surface    = taxes['tau_l_surface']
    tau_l_value      = taxes['tau_l_value']
    tau_L_surface    = taxes['tau_L_surface']
    tau_L_value      = taxes['tau_L_value']
    tau_LR_surface   = taxes['tau_LR_surface']
    tau_LR_value     = taxes['tau_LR_value']
    # tau_LR_0_surface = taxes['tau_LR_0_surface']
    # tau_LR_0_value   = taxes['tau_LR_0_value']
    tau_LDV_sale     = taxes['tau_LDV_sale']
    tau_LR_sale      = taxes['tau_LR_sale']
    tau_H            = taxes['tau_H']
    tau_HI           = taxes['tau_HI']
    tau_D_K          = taxes['tau_D_K']
    tau_D_S          = taxes['tau_D_S']
    tau_D_s          = taxes['tau_D_s']  
    tau_D_H          = taxes['tau_D_H']
    tau_D_h          = taxes['tau_D_h']
    tau_D_L          = taxes['tau_D_L'] 
    tau_D_l          = taxes['tau_D_l'] 
    tau_D_LR         = taxes['tau_D_LR']
    tau_inher_K      = taxes['tau_inher_K']
    tau_inher_S      = taxes['tau_inher_S']
    tau_inher_s      = taxes['tau_inher_s']
    tau_inher_H      = taxes['tau_inher_H']
    tau_inher_h      = taxes['tau_inher_h']
    tau_inher_L      = taxes['tau_inher_L']
    tau_inher_l      = taxes['tau_inher_l']
    tau_inher_LR     = taxes['tau_inher_LR']

    # dilution
    d_K     = n + aleph*tau_inher_K
    d_H     = n + aleph*tau_inher_H
    d_h     = n + aleph*tau_inher_h
    d_S     = n + aleph*tau_inher_S
    d_s     = n + aleph*tau_inher_s
    d_L     = n + aleph*tau_inher_L
    d_l     = n + aleph*tau_inher_l
    d_LR    = n + aleph*tau_inher_LR
    
    
    # solve steady state capital level   
    def res_K(x): 
        return (
        (1+tau_K_inv*(1+n)+d_K*(1+tau_D_K)) 
        - beta*(tau_K_inv*(1-delta_K) + (1-tau_K)*(1-delta_K+env.dK_prod(x)))
        )
    
    k = brentq(res_K, 1e-5, 100)
    
    # per capita capital stock
    K = k*(1+omega)           # *(1+omega)

    # ----- intermediate values -----
    H = env.h_prod(L,S)
    h = env.h_prod(l,s)
    LDV = l + L


    # marginal derivatives
    du_dc = env.du_dc(c,h/omega)
    du_dh = env.du_dh(c,h/omega)
    dU_dC = env.du_dc(C,H)
    dU_dH = env.du_dh(C,H)
    duR_dC_R = env.duR_dc(C_R)
    dH_dL = env.dh_prod_dl(L,S)
    dH_dS = env.dh_prod_ds(L,S)
    dh_dl = env.dh_prod_dl(l,s)
    dh_ds = env.dh_prod_ds(l,s)


    # prices
    R_H_gross = du_dh/du_dc     # using worker FOC
    R_H_net = (1-tau_H)*R_H_gross
    R_K_gross = env.dK_prod(k) + 1 - delta_K
    R_K_net = (1-tau_K)*R_K_gross
    wage = (1-alpha_K)*env.prod(k)
    Y = (1+omega)*env.prod(k)    # total output         (1+omega)

    p_LR = (
        (R_R1*beta-tau_LR_surface)/
        ((1-tau_LR_sale)*(1+n-beta)+tau_LR_value+aleph*tau_inher_LR+d_LR*tau_D_LR)
    )

    # shadow price
    p_LDV = (1+tau_LDV_sale)*p_LR + cost_L1 + cost_L2*n*LDV
    
    
    # if uniformval==1:
    #     # compute implied tax rate on L, l value
    #     tau_L_value = tau_LR_value * p_LR / p_LDV
    #     tau_l_value = tau_LR_value * p_LR / p_LDV
    
    

    # multipliers
    LM_C = dU_dC
    LM_L = LM_C * (1+n-beta) * ((1+tau_LDV_sale)* p_LR + cost_L1 + cost_L2*n*LDV)

    LM_R_C = duR_dC_R
    LM_R_L = (1-tau_LR_sale) * (1+n-beta) * p_LR * LM_R_C
    

    # ----- government revenues -----
    # tax
    T_Ss_inv = tau_S_inv*(n+delta_S)*S + tau_s_inv*(n+delta_S)*s
    T_K_inv = tau_K_inv*(n+delta_K)*K
    T_K     = tau_K*R_K_gross*K
    T_L = (
          (tau_L_surface+tau_L_value*p_LDV)*L 
        + (tau_l_surface+tau_l_value*p_LDV)*l
        +  tau_LDV_sale*p_LR*n*LDV
        )
    T_L_R = (
          varrho*(tau_LR_surface+tau_LR_value*p_LR)*LR_1
        # + varrho*(tau_LR_0_surface+tau_LR_0_value*p_LR)*LR_0
        + tau_LR_sale*p_LR*n*LDV)
    T_H = tau_H*R_H_gross*h + tau_HI*R_H_gross*H
    T_C = (
        varrho * tau_CR * C_R
        + tau_C * C
        + omega * tau_c * c
        + omega * tau_ch * R_H_gross * h
        + tau_CH * R_H_gross * H
    )
    T_C = (
        varrho * tau_CR * C_R
        + tau_C * C
        + omega * tau_c * c
        + omega * tau_ch * R_H_gross * h
        + tau_CH * R_H_gross * H
    )

    # inheritance tax
    T_inher = (
        aleph*tau_inher_K*K + 
        aleph*tau_inher_S*S + 
        aleph*tau_inher_s*s + 
        aleph*tau_inher_H*H*R_H_gross/rho_H +
        aleph*tau_inher_h*h*R_H_net/rho_h +
        aleph*tau_inher_L*p_LDV*L + 
        aleph*tau_inher_l*p_LDV*l
        )
    T_inher_R = varrho*aleph*tau_inher_LR*p_LR*LR_1

    # donation tax
    T_D = (
        d_K*tau_D_K*K + 
        d_S*tau_D_S*S + 
        d_s*tau_D_s*s + 
        d_H*tau_D_H*H*R_H_gross/rho_H +
        d_h*tau_D_h*h*R_H_net/rho_h +
        d_L*tau_D_L*p_LDV*L + 
        d_l*tau_D_l*p_LDV*l
        )
    T_D_R = varrho*d_LR*tau_D_LR*p_LR*LR_1

    # total revenue
    T_total_R = T_L_R + T_inher_R + T_D_R   # in capitalist unit
    T_total_C = T_Ss_inv + T_K_inv + T_K + T_L + T_H + T_inher + T_D

    T_total = T_total_C + T_total_R + T_C

    workers_cons_tax = omega * tau_c * c + omega * tau_ch * R_H_gross * h
    R_net_to_workers_total = T_total - workers_cons_tax
    R_net_to_workers = R_net_to_workers_total / (1 + omega)
    
    
    # national income and GDP
    GDP = Y + R_H_gross*(H+h) + varrho*R_R1*LR_1
    NatIncome_prod = GDP - delta_K*K - delta_S*(S+s)    # GDP net of depreciation
    NatIncome_income = (
        (1+omega)*wage + varrho*R_R1*LR_1 + K*(R_K_gross-1) + R_H_net*(H+h)            # (1+omega)
        + T_total - (T_K_inv + T_K) - delta_S*(S+s))
    NatIncome_demand = (
        varrho*C_R + C + omega*c + R_H_gross*H + R_H_net*h + n*K + n*(S+s) 
        + cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2 + R_H_gross*(H+h) + varrho*R_R1*LR_1)

    # some marginal benefit and cost expressions
    marg_benef_H = (
        beta*(
            dU_dH
            - LM_C*
            (
                (tau_CH + tau_HI)
                + aleph*tau_inher_H/rho_H
                + d_H*tau_D_H/rho_H
            )
            * R_H_gross
        )
    )
    marg_benef_h = (
        beta*LM_C*(1-tau_H)*(1 - aleph*tau_inher_h/rho_h - d_h*tau_D_h/rho_h) * R_H_gross
    )
    tax_wedge_L  = tau_L_surface + (tau_L_value + d_L*tau_D_L + aleph*tau_inher_L)*p_LDV
    tax_wedge_l  = tau_l_surface + (tau_l_value + d_l*tau_D_l + aleph*tau_inher_l)*p_LDV
    
    tax_wedge_S = 1 + tau_S_inv*(1+n) + d_S*(1+tau_D_S)
    tax_wedge_s = 1 + tau_s_inv*(1+n) + d_s*(1+tau_D_s)

    # loss vector
    res = np.zeros(8)
    
    # -------------------------------------------------------------
    # ------------------ steady state conditions ------------------
    # -------------------------------------------------------------

    # ------ capitalist's FOC ------
    # Euler on L
    res[0] = marg_benef_H*dH_dL - LM_C*tax_wedge_L - LM_L

    # Euler on l
    res[1] = marg_benef_h*dh_dl - LM_C*tax_wedge_l - LM_L

    # Euler on S
    res[2] = LM_C*tax_wedge_S - beta*LM_C*(1-delta_S)*(1+tau_S_inv) - marg_benef_H*dH_dS
    
    # Euler on s
    res[3] = LM_C*tax_wedge_s - beta*LM_C*(1-delta_S)*(1+tau_s_inv) - marg_benef_h*dh_ds

    # BCs
    res[4] = (Y - omega*wage + R_H_gross*h
                - (n+delta_K)*K - (n+delta_S)*(S+s) - C
                - n*LDV*p_LR - cost_L1*n*LDV - 1/2*cost_L2*(n*LDV)**2 - T_total_C + R_net_to_workers)    #  + T_total/(1+omega)

    res[5] = (R_R1*LR_1 + R_R0*LR_0 + n*LDV*p_LR/varrho
                - C_R - T_total_R/varrho)

    res[6] = LDV + varrho*LR_1 - kappa/(n+kappa) * Lbar

    # worker
    worker_outlays = (1 + tau_c) * c + (1 + tau_ch) * (R_H_gross/omega) * h
    res[7] = worker_outlays - wage - R_net_to_workers       # - T_total/(1+omega)

    res_worker_intraperiod = (
        (du_dh / (du_dc * R_H_gross)) - ((1 + tau_ch) / (1 + tau_c))
    )

    tau_CH_implied = (
        (tau_C / (1 + tau_C)) * (dU_dH / (dU_dC * R_H_gross))
        if (1 + tau_C) != 0
        else 0.0
    )
    res_tauCH_gap = tau_CH - tau_CH_implied

    # -------------------------------------------------------------
    # ------------------- calibration targets ---------------------
    # -------------------------------------------------------------
    p_LR_seller = p_LR
    
    LR_GDP_ratio    = (p_LR * varrho * (LR_0 + LR_1) / GDP)
    dev_GDP_ratio   = ((p_LDV*(L + l)) / GDP)
    real_estate_GDP_ratio = (R_H_gross/(1-beta) * (H + h) / GDP)
    K_Y_ratio             = (K/Y)
    pLDV_pLR_ratio    = (p_LDV/p_LR_seller)
    LDV_LR_ratio        = ((L+l)/(varrho*(LR_0 + LR_1)))
    
    
    # -------------------------------------------------------------
    # ---------------------- output to keep -----------------------
    # -------------------------------------------------------------
    
    output = {
        # solutions
        'c'  : c,
        'C'  : C, 
        'C_R': C_R,
        's'  : s,
        'S'  : S,
        'l'  : l,
        'L'  : L,
        'LR_0' : LR_0,
        'LR_1' : LR_1,
        'LR' : LR_0 + LR_1,
        
        # prices
        'R_H_gross' :   R_H_gross,
        'R_H_net' :     R_H_net,
        'R_K_gross' :   R_K_gross,
        'R_K_net' :     R_K_net,
        'p_LDV'   :   p_LDV,
        'p_LR'  :   p_LR,
        'wage' :        wage,
        
        # agg variables
        'K' : K,
        'H' : H,
        'h' : h,
        'Y' : Y,
        'LDV'     : LDV,
        'LDV_exp' : LDV*n,
        'GDP' : GDP,
        'NatIncome_prod':   NatIncome_prod,
        'NatIncome_income':  NatIncome_income,
        'NatIncome_demand':  NatIncome_demand,
        'marginal_cost': cost_L1 + cost_L2*n*LDV,
        'total_cost': cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2,
        
        # multipliers
        'LM_C' : LM_C,
        'LM_L' : LM_L,
        'LM_R_C':    LM_R_C,
        'LM_R_L':    LM_R_L,
        
        # taxes
        'T_Ss_inv' : T_Ss_inv,
        'T_K_inv'  : T_K_inv,
        'T_K'      : T_K,
        'T_L'      : T_L,
        'T_L_R'    : T_L_R,
        'T_H'      : T_H,
        'T_C'      : T_C,
        'T_inher'  : T_inher,
        'T_inher_R': T_inher_R,
        'T_D'      : T_D,
        'T_D_R'    : T_D_R,
        'T_total_C' : T_total_C,
        'T_total_R' : T_total_R,
        'T_total'   : T_total,
        'workers_cons_tax': workers_cons_tax,
        'R_net_to_workers_total': R_net_to_workers_total,
        'R_net_to_workers': R_net_to_workers,
        
        # welfare 
        'util_c'   : env.u(c, h/omega),
        'util_C'   : env.u(C, H),
        'util_CR'  : env.uR(C_R),
        'welfare'  : omega*env.u(c,h/env.omega) 
            + gamma_C*env.u(C,H)+varrho*gamma_R*env.uR(C_R),

        # derivatives
        'du_dc'    : du_dc,
        'du_dh'    : du_dh,
        'dU_dC'    : dU_dC,
        'dU_dH'    : dU_dH,
        'duR_dC_R' : duR_dC_R,
        'dH_dL': dH_dL,
        'dH_dS': dH_dS,
        'dh_dl': dh_dl,
        'dh_ds': dh_ds,
        
        # targets
        'LR_GDP_ratio': LR_GDP_ratio,
        'dev_GDP_ratio': dev_GDP_ratio,
        'real_estate_GDP_ratio': real_estate_GDP_ratio,
        'K_Y_ratio': K_Y_ratio,
        'pLDV_pLR_ratio': pLDV_pLR_ratio,
        'LDV_LR_ratio': LDV_LR_ratio,
        'worker_outlays': worker_outlays,
        'res_worker_intraperiod': res_worker_intraperiod,
        'tau_CH_implied': tau_CH_implied,
        'res_tauCH_gap': res_tauCH_gap,
        
        # leftover land
        'leftover': varrho*LR_1 - n*LDV,
        
        # loss
        'Eq_loss': np.sqrt(np.sum(np.square(res)))
    }
    
    # also append the tax values to the output
    output.update(taxes)
    output.update(params_exog)
    output.update(params_endog)
    
    return res, output



def ss_res_binding(params_exog, params_endog, x, taxes):

    # unpack the endogenous variables
    c,C,C_R,s,S,l,Psi_R,Psi_C  = x
    
    # create environment
    env = Environment(params_exog, params_endog, constrained=1)

    # unpack all parameters
    omega, n, aleph, beta = env.omega, env.n, env.aleph, env.beta
    delta_K, delta_S = env.delta_K, env.delta_S
    rho_H, rho_h = env.rho_H, env.rho_h
    alpha_K = env.alpha_K
    varrho = env.varrho
    cost_L1, cost_L2 = env.cost_L1, env.cost_L2
    gamma_C, gamma_R = env.gamma_C, env.gamma_R
    
    kappa = env.kappa   # transformation rate
    eta_b = env.eta_b   # bargaining power
    R_R0, R_R1 = env.R_R0, env.R_R1
    
    # total land
    Lbar = env.Lbar
    LR_0 = (n/(n+kappa) * Lbar) / varrho
    LDV  = (kappa/(n+kappa) * Lbar) / (1+n)
    LR_1 = n * LDV/varrho
    L = LDV - l


    # taxes
    tau_s_inv        = taxes['tau_s_inv']
    tau_S_inv        = taxes['tau_S_inv']  
    tau_K_inv        = taxes['tau_K_inv']
    tau_K            = taxes['tau_K']
    tau_l_surface    = taxes['tau_l_surface']
    tau_l_value      = taxes['tau_l_value']
    tau_L_surface    = taxes['tau_L_surface']
    tau_L_value      = taxes['tau_L_value']
    tau_LR_surface   = taxes['tau_LR_surface']
    tau_LR_value     = taxes['tau_LR_value']
    # tau_LR_0_surface = taxes['tau_LR_0_surface']
    # tau_LR_0_value   = taxes['tau_LR_0_value']
    tau_LDV_sale     = taxes['tau_LDV_sale']
    tau_LR_sale      = taxes['tau_LR_sale']
    tau_H            = taxes['tau_H']
    tau_HI           = taxes['tau_HI']
    tau_D_K          = taxes['tau_D_K']
    tau_D_S          = taxes['tau_D_S']
    tau_D_s          = taxes['tau_D_s']  
    tau_D_H          = taxes['tau_D_H']
    tau_D_h          = taxes['tau_D_h']
    tau_D_L          = taxes['tau_D_L'] 
    tau_D_l          = taxes['tau_D_l'] 
    tau_D_LR         = taxes['tau_D_LR']
    tau_inher_K      = taxes['tau_inher_K']
    tau_inher_S      = taxes['tau_inher_S']
    tau_inher_s      = taxes['tau_inher_s']
    tau_inher_H      = taxes['tau_inher_H']
    tau_inher_h      = taxes['tau_inher_h']
    tau_inher_L      = taxes['tau_inher_L']
    tau_inher_l      = taxes['tau_inher_l']
    tau_inher_LR     = taxes['tau_inher_LR']

    # dilution
    d_K     = n + aleph*tau_inher_K
    d_H     = n + aleph*tau_inher_H
    d_h     = n + aleph*tau_inher_h
    d_S     = n + aleph*tau_inher_S
    d_s     = n + aleph*tau_inher_s
    d_L     = n + aleph*tau_inher_L
    d_l     = n + aleph*tau_inher_l
    d_LR    = n + aleph*tau_inher_LR
    
    
    # solve steady state capital level   
    def res_K(x): 
        return (
        (1+tau_K_inv*(1+n)+d_K*(1+tau_D_K)) 
        - beta*(tau_K_inv*(1-delta_K) + (1-tau_K)*(1-delta_K+env.dK_prod(x)))
        )
    
    k = brentq(res_K, 1e-5, 100)
    
    # per capita capital stock
    K = k*(1+omega)           # *(1+omega)

    # ----- intermediate values -----
    H = env.h_prod(L,S)
    h = env.h_prod(l,s)


    # marginal derivatives
    du_dc = env.du_dc(c,h/omega)
    du_dh = env.du_dh(c,h/omega)
    dU_dC = env.du_dc(C,H)
    dU_dH = env.du_dh(C,H)
    duR_dC_R = env.duR_dc(C_R)
    dH_dL = env.dh_prod_dl(L,S)
    dH_dS = env.dh_prod_ds(L,S)
    dh_dl = env.dh_prod_dl(l,s)
    dh_ds = env.dh_prod_ds(l,s)


    # prices
    R_H_gross = du_dh/du_dc     # using worker FOC
    R_H_net = (1-tau_H)*R_H_gross
    R_K_gross = env.dK_prod(k) + 1 - delta_K
    R_K_net = (1-tau_K)*R_K_gross
    wage = (1-alpha_K)*env.prod(k)
    Y = (1+omega)*env.prod(k)    # total output         (1+omega)


    LM_R_C = duR_dC_R
    
    # land price
    p_LR = (
        (R_R1*beta-tau_LR_surface + (1+n) * Psi_R/LM_R_C)/
        ((1-tau_LR_sale)*(1+n-beta)+tau_LR_value+aleph*tau_inher_LR+d_LR*tau_D_LR)
    )
    p_LR_seller = (
        (R_R1*beta-tau_LR_surface)/
        ((1-tau_LR_sale)*(1+n-beta)+tau_LR_value+aleph*tau_inher_LR+d_LR*tau_D_LR)
    )

    # shadow price
    p_LDV = (1+tau_LDV_sale)*p_LR + cost_L1 + cost_L2*n*LDV
    
    # if uniformval==1:
    #     # compute implied tax rate on L, l value
    #     tau_L_value = tau_LR_value * p_LR / p_LDV
    #     tau_l_value = tau_LR_value * p_LR / p_LDV

    # multipliers
    LM_C = dU_dC
    LM_L = LM_C * (1+n-beta) * ((1+tau_LDV_sale)* p_LR + cost_L1 + cost_L2*n*LDV) + Psi_C * (1+n-beta)

    LM_R_L = (1-tau_LR_sale) * (1+n-beta) * p_LR * LM_R_C - (1+n-beta)*Psi_R
    
    p_LR_buyer = (
        (LM_L/LM_C)/(1 + n - beta) - cost_L1 - cost_L2*n*LDV
    ) / (1 + tau_LDV_sale)

    # ----- government revenues -----
    # tax
    T_Ss_inv = tau_S_inv*(n+delta_S)*S + tau_s_inv*(n+delta_S)*s
    T_K_inv = tau_K_inv*(n+delta_K)*K
    T_K     = tau_K*R_K_gross*K
    T_L = (
          (tau_L_surface+tau_L_value*p_LDV)*L 
        + (tau_l_surface+tau_l_value*p_LDV)*l
        +  tau_LDV_sale*p_LR*n*LDV
        )
    T_L_R = (
          varrho*(tau_LR_surface+tau_LR_value*p_LR)*LR_1
        # + varrho*(tau_LR_0_surface+tau_LR_0_value*p_LR)*LR_0
        + tau_LR_sale*p_LR*n*LDV)
    T_H = tau_H*R_H_gross*h + tau_HI*R_H_gross*H

    # inheritance tax
    T_inher = (
        aleph*tau_inher_K*K + 
        aleph*tau_inher_S*S + 
        aleph*tau_inher_s*s + 
        aleph*tau_inher_H*H*R_H_gross/rho_H +
        aleph*tau_inher_h*h*R_H_net/rho_h +
        aleph*tau_inher_L*p_LDV*L + 
        aleph*tau_inher_l*p_LDV*l
        )
    T_inher_R = varrho*aleph*tau_inher_LR*p_LR*LR_1

    # donation tax
    T_D = (
        d_K*tau_D_K*K + 
        d_S*tau_D_S*S + 
        d_s*tau_D_s*s + 
        d_H*tau_D_H*H*R_H_gross/rho_H +
        d_h*tau_D_h*h*R_H_net/rho_h +
        d_L*tau_D_L*p_LDV*L + 
        d_l*tau_D_l*p_LDV*l
        )
    T_D_R = varrho*d_LR*tau_D_LR*p_LR*LR_1

    # total revenue
    T_total_R = T_L_R + T_inher_R + T_D_R   # in capitalist unit
    T_total_C = T_Ss_inv + T_K_inv + T_K + T_L + T_H + T_inher + T_D

    T_total = T_total_C + T_total_R + T_C

    workers_cons_tax = omega * tau_c * c + omega * tau_ch * R_H_gross * h
    R_net_to_workers_total = T_total - workers_cons_tax
    R_net_to_workers = R_net_to_workers_total / (1 + omega)
    
    
    # national income and GDP
    GDP = Y + R_H_gross*(H+h) + varrho*(R_R1*LR_1 + R_R0*LR_0)
    NatIncome_prod = GDP - delta_K*K - delta_S*(S+s)    # GDP net of depreciation
    NatIncome_income = (
        (1+omega)*wage + varrho*R_R1*LR_1 + K*(R_K_gross-1) + R_H_net*(H+h)            # (1+omega)
        + T_total - (T_K_inv + T_K) - delta_S*(S+s))
    NatIncome_demand = (
        varrho*C_R + C + omega*c + R_H_gross*H + R_H_net*h + n*K + n*(S+s) 
        + cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2 + R_H_gross*(H+h) + varrho*R_R1*LR_1 + varrho*R_R0*LR_0)

    # some marginal benefit and cost expressions
    marg_benef_H = (
        beta*(
            dU_dH
            - LM_C*
            (
                (tau_CH + tau_HI)
                + aleph*tau_inher_H/rho_H
                + d_H*tau_D_H/rho_H
            )
            * R_H_gross
        )
    )
    marg_benef_h = (
        beta*LM_C*(1-tau_H)*(1 - aleph*tau_inher_h/rho_h - d_h*tau_D_h/rho_h) * R_H_gross
    )
    tax_wedge_L  = tau_L_surface + (tau_L_value + d_L*tau_D_L + aleph*tau_inher_L)*p_LDV
    tax_wedge_l  = tau_l_surface + (tau_l_value + d_l*tau_D_l + aleph*tau_inher_l)*p_LDV
    
    tax_wedge_S = 1 + tau_S_inv*(1+n) + d_S*(1+tau_D_S)
    tax_wedge_s = 1 + tau_s_inv*(1+n) + d_s*(1+tau_D_s)

    # loss vector
    res = np.zeros(8)
    
    # -------------------------------------------------------------
    # ------------------ steady state conditions ------------------
    # -------------------------------------------------------------

    # ------ capitalist's FOC ------
    # Euler on L
    res[0] = marg_benef_H*dH_dL - LM_C*tax_wedge_L - LM_L

    # Euler on l
    res[1] = marg_benef_h*dh_dl - LM_C*tax_wedge_l - LM_L

    # Euler on S
    res[2] = LM_C*tax_wedge_S - beta*LM_C*(1-delta_S)*(1+tau_S_inv) - marg_benef_H*dH_dS
    
    # Euler on s
    res[3] = LM_C*tax_wedge_s - beta*LM_C*(1-delta_S)*(1+tau_s_inv) - marg_benef_h*dh_ds

    # BCs
    res[4] = (Y - omega*wage + R_H_gross*h
                - (n+delta_K)*K - (n+delta_S)*(S+s) - C
                - n*LDV*p_LR - cost_L1*n*LDV - 1/2*cost_L2*(n*LDV)**2 - T_total_C + R_net_to_workers)    #  + T_total/(1+omega)

    res[5] = (R_R1*LR_1 + R_R0*LR_0 + n*LDV*p_LR/varrho
                - C_R - T_total_R/varrho)

    # worker
    worker_outlays = (1 + tau_c) * c + (1 + tau_ch) * (R_H_gross/omega) * h
    res[6] = worker_outlays - wage - R_net_to_workers       # - T_total/(1+omega)

    res_worker_intraperiod = (
        (du_dh / (du_dc * R_H_gross)) - ((1 + tau_ch) / (1 + tau_c))
    )

    tau_CH_implied = (
        (tau_C / (1 + tau_C)) * (dU_dH / (dU_dC * R_H_gross))
        if (1 + tau_C) != 0
        else 0.0
    )
    res_tauCH_gap = tau_CH - tau_CH_implied
    
    # multiplier
    res[7] = (eta_b*p_LR_seller + (1-eta_b)*p_LR_buyer - p_LR)      # bargaining
    
    
    # -------------------------------------------------------------
    # ------------------- calibration targets ---------------------
    # -------------------------------------------------------------
    imputed_value_LR_0 = beta * R_R0 / (1+n-beta)
    LR_GDP_ratio    = (p_LR * varrho * LR_1 + imputed_value_LR_0* varrho * LR_0 / GDP)
    dev_GDP_ratio   = ((p_LDV*(L + l)) / GDP)
    real_estate_GDP_ratio = (R_H_gross/(1-beta) * (H + h) / GDP)
    K_Y_ratio             = (K/Y)
    pLDV_pLR_ratio    = (p_LDV/p_LR_seller)
    LDV_LR_ratio        = ((L+l)/(varrho*(LR_0 + LR_1)))
    
    # -------------------------------------------------------------
    # ---------------------- output to keep -----------------------
    # -------------------------------------------------------------
    output = {
        # solutions
        'c'  : c,
        'C'  : C, 
        'C_R': C_R,
        's'  : s,
        'S'  : S,
        'l'  : l,
        'L'  : L,
        'LR' : LR_0 + LR_1,
        'LR_0' : LR_0,
        'LR_1' : LR_1,
        
        # prices
        'R_H_gross' :   R_H_gross,
        'R_H_net' :     R_H_net,
        'R_K_gross' :   R_K_gross,
        'R_K_net' :     R_K_net,
        'p_LDV'    :   p_LDV,
        'p_LR'     :   p_LR,
        'p_LR_buyer' : p_LR_buyer,
        'p_LR_seller': p_LR_seller,
        'wage' :        wage,
        
        # agg variables
        'K' : K,
        'H' : H,
        'h' : h,
        'Y' : Y,
        'LDV'     : LDV,
        'LDV_exp' : LDV*n,
        'GDP' : GDP,
        'NatIncome_prod':   NatIncome_prod,
        'NatIncome_income':  NatIncome_income,
        'NatIncome_demand':  NatIncome_demand,
        'marginal_cost': cost_L1 + cost_L2*n*LDV,
        'total_cost': cost_L1*n*LDV + 1/2*cost_L2*(n*LDV)**2,
        
        # multipliers
        'LM_C' : LM_C,
        'LM_L' : LM_L,
        'LM_R_C':    LM_R_C,
        'LM_R_L':    LM_R_L,
        'Psi_C' :    Psi_C,
        'Psi_R' :    Psi_R,
        
        # taxes
        'T_Ss_inv' : T_Ss_inv,
        'T_K_inv'  : T_K_inv,
        'T_K'      : T_K,
        'T_L'      : T_L,
        'T_L_R'    : T_L_R,
        'T_H'      : T_H,
        'T_C'      : T_C,
        'T_inher'  : T_inher,
        'T_inher_R': T_inher_R,
        'T_D'      : T_D,
        'T_D_R'    : T_D_R,
        'T_total_C' : T_total_C,
        'T_total_R' : T_total_R,
        'T_total'   : T_total,
        'workers_cons_tax': workers_cons_tax,
        'R_net_to_workers_total': R_net_to_workers_total,
        'R_net_to_workers': R_net_to_workers,
        
        # welfare 
        'util_c'   : env.u(c, h/omega),
        'util_C'   : env.u(C, H),
        'util_CR'  : env.uR(C_R),
        'welfare'  : omega*env.u(c,h/env.omega) 
            + gamma_C*env.u(C,H)+varrho*gamma_R*env.uR(C_R),

        # derivatives
        'du_dc'    : du_dc,
        'du_dh'    : du_dh,
        'dU_dC'    : dU_dC,
        'dU_dH'    : dU_dH,
        'duR_dC_R' : duR_dC_R,
        'dH_dL': dH_dL,
        'dH_dS': dH_dS,
        'dh_dl': dh_dl,
        'dh_ds': dh_ds,
        

        # targets
        'LR_GDP_ratio': LR_GDP_ratio,
        'dev_GDP_ratio': dev_GDP_ratio,
        'real_estate_GDP_ratio': real_estate_GDP_ratio,
        'K_Y_ratio': K_Y_ratio,
        'pLDV_pLR_ratio': pLDV_pLR_ratio,
        'LDV_LR_ratio': LDV_LR_ratio,
        'worker_outlays': worker_outlays,
        'res_worker_intraperiod': res_worker_intraperiod,
        'tau_CH_implied': tau_CH_implied,
        'res_tauCH_gap': res_tauCH_gap,
        
        # loss
        'Eq_loss': np.sqrt(np.sum(np.square(res))),
        'binding' : 1
    }
    
    # also append the tax values to the output
    output.update(taxes)
    output.update(params_exog)
    output.update(params_endog)
    
    return res, output








#####################################################################################
######################## Simulation and Plotting Tools ##############################
#####################################################################################


# -------------------------------------------------------
# ---------------- Create Tables/latex ------------------
# -------------------------------------------------------

name_symb1 = [                      # consumption
    ('Capitalist cons.', 'C', r'$C$'),
    ('Worker cons. (per worker)', 'c', r'$c$'),
    ('Rentier cons. (per rentier)', 'C_R', r'$C^{R}$'),
    ('Home-owner land', 'L', r'$\mathcal{L}$'),
    ('Rental land', 'l', r'$l$'),
    ('Developed land', 'LDV', r'$\mathcal{L}+l$'),
    ('Raw land (per rentier)', 'LR', r'$\mathcal{L}^{R}$'),
    ('Home-owner housing', 'H', r'$H$'),
    ('Rental housing', 'h', r'$h$')
]

name_symb2 = [                      # production and stocks
    ('Total output', 'Y', r'$Y$'),
    ('Capital', 'K', r'$K$'),
    ('Home-owner struct.', 'S', r'$S$'),
    ('Rental struct.', 's', r'$s$'),
    ('GDP', 'GDP', r'GDP'),
    ('National income','NatIncome_prod', ''),
]

name_symb3 = [                      # prices
    ('Price of raw land', 'p_LR', r'$p^{\mathcal{L}^{R}}$'),
    ('Price of developed land', 'p_LDV', r'$p^{\mathcal{L}^{dev}}$'),
    ('Wage (per worker)', 'wage', r'$w$'),
    ('Gross return on capital', 'R_K_gross', r'$R^{Kgross}$'),
    ('Gross rental rate', 'R_H_gross', r'$R^{Hgross}$'),
]

name_symb4 = [                        # targets/wealth component
    ('Capital/GDP', 'K_Y_ratio', r'$K/Y$'),
    ('Raw land value/GDP', 'LR_GDP_ratio', r'$\frac{p^{\mathcal{L}^{R}}\mathcal{L}^{R}}{GDP}$'),
    ('Dev land value/GDP', 'dev_GDP_ratio', r'$\frac{p^{\mathcal{L}^{dev}}\mathcal{L}^{dev}}{GDP}$'),
    ('Real estate value/GDP', 'real_estate_GDP_ratio', r'$\frac{\frac{R^{Hgross}}{1-\beta}(H+h)}{GDP}$'),
    ('Land size ratio', 'LDV_LR_ratio', r'$\mathcal{L}^{dev}/(\varrho \mathcal{L}^{R})$'),
    ('Land price ratio', 'pLDV_pLR_ratio', r'$p^{\mathcal{L}^{dev}}/p^{\mathcal{L}^{R}}$'),
]

name_symb5 = [                          # welfare
    ('Worker utility', 'util_c', r'$u_{c}$'),
    ('Capitalist utility', 'util_C', r'$U_{C}$'),
    ('Rentier utility', 'util_CR', r'$U_{R}$'),
    ('Total welfare', 'welfare', ''),     # r'$\omega u_{c} + \gamma_C U_{C} + \gamma_R U_{R}$'
]

name_symb_ls = [name_symb1, name_symb2, name_symb3, name_symb4, name_symb5]



def create_table(sol_calib, sol_taxeq0, targets):
    
    # create table
    tb = PrettyTable()
    tb.field_names = ['Name', 'Symbol', 'Calibrated', 'Laissez faire', 'Target']
    
    # add rows to table by name_symb
    for name_symb in name_symb_ls:
        # create raw table as a list to transform into PrettyTable
        raw_table = []
        
        for name, symb, _ in name_symb:
            
            # only fill in Target column when there is one
            if symb in targets.keys():
                target = targets[symb]
            else:
                target = '-'
            raw_table.append([name, symb, sol_calib[symb], sol_taxeq0[symb], target])
            
        tb.add_rows(
            raw_table
        )
        # add a divider
        tb.add_row(['-----------------'] * 5)
    print(tb)
    

def create_rows(n_col, ls):
    n_row = divmod(len(ls), n_col)[0] + (divmod(len(ls), n_col)[1]>0)   # number of rows

    # store strings in a 2d np array
    tb = np.full([n_row, n_col], fill_value='', dtype=object)

    # fill first col, and then second col, and so on
    for i in range(len(ls)):
        col_indx, row_indx = divmod(i, n_row)
        tb[row_indx, col_indx] = ls[i]

    rows = []
    for i in range(n_row):
        rows.append(tb[i,:][0] + ' & ' + tb[i,:][1])
    return rows
    
    
def latex_table(sol_calib, sol_taxeq0, targets):
    
    # num of columns
    n_col = 2
    
    # round off all solutions to rd decimals
    rd = 2
    sol_calib   = {key: round(sol_calib[key], rd) for key in sol_calib}
    sol_taxeq0  = {key: round(sol_taxeq0[key], rd) for key in sol_taxeq0}
    targets     = {key: round(targets[key], rd) for key in targets}
    
    # some convenient command
    tnl = r"\tabularnewline"
    br = r" & "
    nl = '\n '
    
    # front matter of a table
    front = r"""
\begin{landscape}
\begin{table}
\begin{centering}
\begin{tabular}{lllllllllll}
\hline 
Name  & Symbol  & Calib & Laissez faire & Target &  Name  & Symbol  & Calib & Laissez faire & Target \tabularnewline
\hline
"""
    
    # end matter of a table
    end = r"""
\hline
\multicolumn{9}{l}{\textit{Note: }All variables are represented in per-capitalist terms,
with exceptions specified in brackets. We use \* t0 denote per-worker values, and \** to denote per-rentier values.}\tabularnewline
\hline
\end{tabular}
\caption{Column "Calib" presents the calibrated economy when the tax on capital 
 ($\tau_{K}=1.1\%$) is set to 1.1\%. Column "Laissez faire" presents the same economy but with all taxes set 
to zero.}
\end{centering}
\end{table}
\end{landscape}
"""

    tex_output = front
    
    # ------------- create table section by section ------------
    # Each row of a "table column" consists of 5 items: name, symbol, value (calib), value (Laissez faire), target
    # We put two table columns, separated by a break in the middle
    section_name_ls = ['Consumption', 'Production and stocks', 'Prices', 'Wealth component', 'Welfare']

    for i in range(len(name_symb_ls)):
        
        name_symb = name_symb_ls[i]
        section_name = section_name_ls[i]
        
        tex_output = tex_output + r"\textit{" + f"{section_name}." + r"}" + br*10 + tnl + nl
        
        # first create rows of tex
        rows_temp = []
        for name, symb, symb_tex in name_symb:
            
            # get target if there is one, else set to '-'
            if symb in targets.keys():
                target = fr"${targets[symb]}$"
            else:
                target = ''
            
            # create a tex row of five items
            line = (
                name + br + 
                symb_tex + br + 
                fr"${sol_calib[symb]}$" + br + 
                fr"${sol_taxeq0[symb]}$" + br + 
                target
            )
            rows_temp.append(line)
            
        # next sort rows into two columns
        rows = create_rows(n_col, rows_temp)
        
        # finally add row by row to tex 
        tex_output = tex_output + (tnl + nl).join(rows) + (tnl + nl) + br*10 + (tnl + nl)
     
    tex_output = tex_output + end
    
    return tex_output


# -------------------------------------------------------
# ------------- Simulate steady state tax ---------------
# -------------------------------------------------------
def comp_stat_tax(params_exog: dict, params_endog: dict, 
                  taxes0: dict, tax_dict: dict, init_guess: dict, constrained=0, uniformval=0):
    
    taxes2var = taxes0.copy()       # make a copy of initial tax to loop over
    
    # store all variables and then convert to a pandas dataframe
    output_path = []
    
    
    for i in range(len(list(tax_dict.values())[0])):
        
        for key in tax_dict.keys():
            taxes2var[key] = tax_dict[key][i]
            
        if constrained==0:
            output = solve_ss(params_exog, params_endog, taxes2var, init_guess, uniformval=uniformval)
        else:
            try:
                output = solve_ss_constructible(params_exog, params_endog, taxes2var, init_guess)
            except ValueError:
                output['sol_flag'] = 1      # assign invalid solution
        output_path.append(output)
        
        # update initial guess from last iteration's solution
        init_guess = output
           
    df = pd.DataFrame.from_records(output_path)
    
    return df
    

# -------------------------------------------------------
# ----------- Inspect price and consumption -------------
# -------------------------------------------------------

def inspect_vars(tax_path):
    
    fig, axs = plt.subplots(3, 3, figsize=[10,5])
    axs[0,0].plot(tax_path['C'], label='C')
    axs[0,1].plot(tax_path['c'], label='c')
    axs[0,2].plot(tax_path['C_R'], label='C_R')
    axs[1,0].plot(tax_path['p_LR'], label='p_LR')
    axs[1,1].plot(tax_path['p_LDV'], label='p_LDV')
    axs[1,2].plot(tax_path['K'], label='K')
    axs[2,0].plot(tax_path['L'], label='L')
    axs[2,1].plot(tax_path['l'], label='l')
    axs[2,2].plot(tax_path['LR'], label='LR')
    
    for i in [0,1, 2]:
        for j in [0,1,2]:
            # axs[i,j].set_ylim([-1,10])
            axs[i,j].legend()
            valid_reg = tax_path['sol_flag'][tax_path['sol_flag']==0]
            axs[i,j].axvspan(valid_reg.index[0], valid_reg.index[-1], color='yellow', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    