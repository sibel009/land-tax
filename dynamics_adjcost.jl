using LinearAlgebra
using BenchmarkTools
# using Optim
using StaticArrays

using LeastSquaresOptim
using ForwardDiff

using JSON
using DelimitedFiles
using Plots

########################################################
########################################################
# ---------------------- functions ---------------------

# useful functions
u(c, h, α_C, σ)     = ((c^α_C)*(h^(1-α_C)))^(1-σ) / (1-σ)
uR(c, σ)            = c^(1-σ) / (1-σ)
duR_dc(c, σ)        = c^(-σ)
du_dc(c, h, α_C, σ) = (α_C*(c^(α_C-1))*(h^(1-α_C)))*((c^α_C)*h^(1-α_C))^(-σ)
du_dh(c, h, α_C, σ) = ((1-α_C)*(c^α_C)*(h^(-α_C)))*(((c^α_C)*(h^(1-α_C)))^(-σ))

h_prod(a_L, l, s)       = (l^a_L)*(s^(1-a_L))
dh_prod_dl(a_L, l, s)   = a_L*(l^(a_L-1))*(s^(1-a_L))
dh_prod_ds(a_L, l, s)   = (1-a_L)*(l^a_L)*(s^(-a_L))

prod(α_K, A, k)         = A*(k)^(α_K)
dK_prod(α_K, A, k)      = (α_K)*A*(k)^(α_K-1.0)

# convex adjustment cost 
adj_cost(I, Iss, b_cost)        = b_cost/2 * (I - Iss)^2
marg_adj_cost(I, Iss, b_cost)   = b_cost * (I - Iss)




function transition_loss!(loss, x, init_cond, terminal_cond, τ_path, T, params)

    # xx is of length nvars*(T+2)
    xx = [init_cond; x; terminal_cond]

    for t in 1:(T+1)
        transition_interp_loss!(loss, xx, τ_path, t, params)
    end
    
end

function transition_interp_loss!(loss, xx, τ_path, t, params)

    # unpack parameters
    (; ℵ, n, ω, ϱ, σ, β, A, δ_k, δ_s, 
        α_C, α_K, a_L, cost_L1, cost_L2, Lbar, ρ_h, ρ_H, R_R, nvars) = params

    # unpack variables
    Km1 = xx[nvars*(t-1) + 1]
    Sm1 = xx[nvars*(t-1) + 2]
    sm1 = xx[nvars*(t-1) + 3]

    K0 = xx[nvars*(t) + 1]
    S0 = xx[nvars*(t) + 2]
    s0 = xx[nvars*(t) + 3]
    L0 = xx[nvars*(t) + 4]
    l0 = xx[nvars*(t) + 5]
    c0 = xx[nvars*(t) + 6]
    C0 = xx[nvars*(t) + 7]
    C_R0  = xx[nvars*(t) + 8]
    p_LR0 = xx[nvars*(t) + 9]


    K1 = xx[nvars*(t+1) + 1]
    S1 = xx[nvars*(t+1) + 2]
    s1 = xx[nvars*(t+1) + 3]
    L1 = xx[nvars*(t+1) + 4]
    l1 = xx[nvars*(t+1) + 5]
    c1 = xx[nvars*(t+1) + 6]
    C1 = xx[nvars*(t+1) + 7]
    C_R1  = xx[nvars*(t+1) + 8]
    p_LR1 = xx[nvars*(t+1) + 9]

    K2 = xx[nvars*(t+2) + 1]            # for adjustment cost
    S2 = xx[nvars*(t+2) + 2]
    s2 = xx[nvars*(t+2) + 3]
    L2 = xx[nvars*(t+2) + 4]
    l2 = xx[nvars*(t+2) + 5]


    # taxes today
    τ_s_inv0    = τ_path[t, nτ_s_inv]                 # capital, structure taxes
    τ_S_inv0    = τ_path[t, nτ_S_inv]     
    τ_K_inv0    = τ_path[t, nτ_K_inv]
    τ_K0        = τ_path[t, nτ_K]
 
    τ_l_surf0   = τ_path[t, nτ_l_surf]                 # land taxes 
    τ_l_val0    = τ_path[t, nτ_l_val]
    τ_L_surf0   = τ_path[t, nτ_L_surf]
    τ_L_val0    = τ_path[t, nτ_L_val]
    τ_LR_surf0  = τ_path[t, nτ_LR_surf]
    τ_LR_val0   = τ_path[t, nτ_LR_val]
    τ_LDV_sale0 = τ_path[t, nτ_LDV_sale]
    τ_LR_sale0  = τ_path[t, nτ_LR_sale]
    
    τ_H0    = τ_path[t, nτ_H]
    τ_HI0   = τ_path[t, nτ_HI]                     # housing taxes
 
    τ_D_K0  = τ_path[t, nτ_D_K]                     # donation taxes
    τ_D_S0  = τ_path[t, nτ_D_S]
    τ_D_s0  = τ_path[t, nτ_D_s]
    τ_D_H0  = τ_path[t, nτ_D_H]
    τ_D_h0  = τ_path[t, nτ_D_h]
    τ_D_L0  = τ_path[t, nτ_D_L]
    τ_D_l0  = τ_path[t, nτ_D_l]
    τ_D_LR0 = τ_path[t, nτ_D_LR]
 
    τ_ℵ_K0  = τ_path[t, nτ_ℵ_K]              # ℵitance taxes
    τ_ℵ_S0  = τ_path[t, nτ_ℵ_S]
    τ_ℵ_s0  = τ_path[t, nτ_ℵ_s]
    τ_ℵ_H0  = τ_path[t, nτ_ℵ_H]
    τ_ℵ_h0  = τ_path[t, nτ_ℵ_h]
    τ_ℵ_L0  = τ_path[t, nτ_ℵ_L]
    τ_ℵ_l0  = τ_path[t, nτ_ℵ_l]
    τ_ℵ_LR0 = τ_path[t, nτ_ℵ_LR]

    d_K0     = n + ℵ*τ_ℵ_K0             # dilution
    d_H0     = n + ℵ*τ_ℵ_H0
    d_h0     = n + ℵ*τ_ℵ_h0
    d_S0     = n + ℵ*τ_ℵ_S0
    d_s0     = n + ℵ*τ_ℵ_s0
    d_L0     = n + ℵ*τ_ℵ_L0
    d_l0     = n + ℵ*τ_ℵ_l0
    d_LR0    = n + ℵ*τ_ℵ_LR0

    # taxes tomorrow
    τ_s_inv1    = τ_path[t+1, nτ_s_inv]                 # capital, structure taxes
    τ_S_inv1    = τ_path[t+1, nτ_S_inv]     
    τ_K_inv1    = τ_path[t+1, nτ_K_inv]
    τ_K1        = τ_path[t+1, nτ_K]
 
    τ_LDV_sale1 = τ_path[t+1, nτ_LDV_sale]                # land taxes 
    τ_LR_sale1  = τ_path[t+1, nτ_LR_sale]
    
    τ_H1    = τ_path[t+1, nτ_H]                    # housing taxes
    τ_HI1   = τ_path[t+1, nτ_HI]
    τ_D_H1  = τ_path[t+1, nτ_D_H]
    τ_D_h1  = τ_path[t+1, nτ_D_h]

    τ_ℵ_H1  = τ_path[t, nτ_ℵ_H]
    τ_ℵ_h1  = τ_path[t, nτ_ℵ_h]
    d_H1     = n + ℵ*τ_ℵ_H1
    d_h1     = n + ℵ*τ_ℵ_h1

    # ------------ derived variables ------------


    # adjustment cost 
    # K_ss0 = xx[end-8]
    # S_ss0 = xx[end-7]
    # s_ss0 = xx[end-6]

    # Inv_K_ss0 = (n + δ_k)*K_ss0
    # Inv_S_ss0 = (n + δ_s)*S_ss0
    # Inv_s_ss0 = (n + δ_s)*s_ss0

    # Inv_Km1 = K0*(1+n) - Km1*(1-δ_k)
    # Inv_Sm1 = S0*(1+n) - Sm1*(1-δ_s)
    # Inv_sm1 = s0*(1+n) - sm1*(1-δ_s)

    Inv_K0_star = K0*(n+δ_k)
    Inv_S0_star = S0*(n+δ_s)
    Inv_s0_star = s0*(n+δ_s)

    Inv_K0 = K1*(1+d_K0) - K0*(1-δ_k)
    Inv_S0 = S1*(1+d_S0) - S0*(1-δ_s)
    Inv_s0 = s1*(1+d_s0) - s0*(1-δ_s)

    Inv_K1_star = K1*(n+δ_k)
    Inv_S1_star = S1*(n+δ_s)
    Inv_s1_star = s1*(n+δ_s)

    Inv_K1 = K2*(1+n) - K1*(1-δ_k)
    Inv_S1 = S2*(1+n) - S1*(1-δ_s)
    Inv_s1 = s2*(1+n) - s1*(1-δ_s)

    adj_cost_K0 = adj_cost(Inv_K0, Inv_K0_star, 4)
    adj_cost_S0 = adj_cost(Inv_S0, Inv_S0_star, 2)
    adj_cost_s0 = adj_cost(Inv_s0, Inv_s0_star, 2)

    marg_adj_cost_K0 = marg_adj_cost(Inv_K0, Inv_K0_star, 4)
    marg_adj_cost_S0 = marg_adj_cost(Inv_S0, Inv_S0_star, 2)
    marg_adj_cost_s0 = marg_adj_cost(Inv_s0, Inv_s0_star, 2)

    marg_adj_cost_K1 = marg_adj_cost(Inv_K1, Inv_K1_star, 4)
    marg_adj_cost_S1 = marg_adj_cost(Inv_S1, Inv_S1_star, 2)
    marg_adj_cost_s1 = marg_adj_cost(Inv_s1, Inv_s1_star, 2)

    # land
    LDV0 = L0 + l0
    LDV1 = L1 + l1
    LDV2 = L2 + l2
    LR0 = (Lbar - LDV0)/ϱ
    LR1 = (Lbar - LDV1)/ϱ

    # adjustment cost of land
    marg_cost_L0 = cost_L1 + cost_L2*((1+n)*LDV1 - LDV0)
    marg_cost_L1 = cost_L1 + cost_L2*((1+n)*LDV2 - LDV1)
    total_cost = cost_L1 * ((1+n)*LDV1 - LDV0) + 1/2 * cost_L2 * ((1+n)*LDV1 - LDV0)^2

    # housing, capital, marginal utilities
    h0 = h_prod(a_L, l0, s0)
    H0 = h_prod(a_L, L0, S0)
    k0 = K0/(1 + ω)
    
    du_dc0 = du_dc(c0, h0/ω, α_C, σ)
    du_dh0 = du_dh(c0, h0/ω, α_C, σ)
    duR_dC_R0 = duR_dc(C_R0, σ)
    dU_dC0 = du_dc(C0, H0, α_C, σ)
    dU_dH0 = du_dh(C0, H0, α_C, σ)
    duR_dC_R0 = duR_dc(C_R0, σ)
    
    
    h1 = h_prod(a_L, l1, s1)
    H1 = h_prod(a_L, L1, S1)
    k1 = K1/(1 + ω)
    
    du_dc1 = du_dc(c1, h1/ω, α_C, σ)
    du_dh1 = du_dh(c1, h1/ω, α_C, σ)
    duR_dC_R1 = duR_dc(C_R1, σ)
    dU_dC1 = du_dc(C1, H1, α_C, σ)
    dU_dH1 = du_dh(C1, H1, α_C, σ)
    duR_dC_R1 = duR_dc(C_R1, σ)
    
    
    # marginal product of housing
    du_dc0 = du_dc(c0, h0/ω, α_C, σ)
    du_dh0 = du_dh(c0, h0/ω, α_C, σ)

    dU_dC0 = du_dc(C0, H0, α_C, σ)
    dU_dH0 = du_dh(C0, H0, α_C, σ)
    duR_dC_R0 = duR_dc(C_R0, σ)
    
    du_dc1 = du_dc(c1, h1/ω, α_C, σ)
    du_dh1 = du_dh(c1, h1/ω, α_C, σ)
    dh_dl1 = dh_prod_dl(a_L, l1, s1)
    dh_ds1 = dh_prod_ds(a_L, l1, s1)

    dU_dC1 = du_dc(C1, H1, α_C, σ)
    dU_dH1 = du_dh(C1, H1, α_C, σ)
    dH_dL1 = dh_prod_dl(a_L, L1, S1)
    dH_dS1 = dh_prod_ds(a_L, L1, S1)
    duR_dC_R1 = duR_dc(C_R1, σ)


    # multipliers and prices, and some marginal benefit and marginal cost expressions
    LM_C0 = dU_dC0
    LM_C1 = dU_dC1
    LM_R_C0 = duR_dC_R0
    LM_R_C1 = duR_dC_R1
    
    R_H_gross0 = du_dh0/du_dc0
    R_H_gross1 = du_dh1/du_dc1
    R_H_net0 = (1-τ_H0)*R_H_gross0
    
    R_K_gross0 = dK_prod(α_K, A, k0) + 1 - δ_k
    R_K_gross1 = dK_prod(α_K, A, k1) + 1 - δ_k
    R_K_net1 = (1-τ_K1)*R_K_gross1
    
    wage0 = (1-α_K)*prod(α_K, A, k0)
    
    Y0 = (1+ω)*prod(α_K, A, k0)
    
    p_LDV0 = (1 + τ_LDV_sale0)*p_LR0 + marg_cost_L0
    
    marg_benef_H = (
        β*(dU_dH1 - LM_C1*(τ_HI1 + ℵ*τ_ℵ_H1/ρ_H + d_H1*τ_D_H1/ρ_H)*R_H_gross1)
    )
    marg_benef_h = (
        β*LM_C1*(1-τ_H1)*(1 - ℵ*τ_ℵ_h1/ρ_h - d_h1*τ_D_h1/ρ_h) * R_H_gross1
    )
    tax_wedge_L  = τ_L_surf0 + (τ_L_val0 + d_L0*τ_D_L0 + ℵ*τ_ℵ_L0)*p_LDV0
    tax_wedge_l  = τ_l_surf0 + (τ_l_val0 + d_l0*τ_D_l0 + ℵ*τ_ℵ_l0)*p_LDV0
    tax_wedge_LR = τ_LR_surf0 + (τ_LR_val0 + d_LR0*τ_D_LR0 + ℵ*τ_ℵ_LR0)*p_LR0
    
    tax_wedge_S = 1 + τ_S_inv0*(1+n) + d_S0*(1+τ_D_S0) + marg_adj_cost_S0*(1+n)
    tax_wedge_s = 1 + τ_s_inv0*(1+n) + d_s0*(1+τ_D_s0) + marg_adj_cost_s0*(1+n)
    tax_wedge_K = 1 + τ_K_inv0*(1+n) + d_K0*(1+τ_D_K0) + marg_adj_cost_K0*(1+n)
    
    LM_L0   = LM_C0*((1+n)*(1+τ_LDV_sale0)*p_LR0 + marg_cost_L0*(1+n)) - β*LM_C1*((1+τ_LDV_sale1)*p_LR1 + marg_cost_L1)
    LM_R_L0 = LM_R_C0*(1+n)*(1-τ_LR_sale0)*p_LR0 - β*LM_R_C1*(1-τ_LR_sale1)*p_LR1
    

    
    # -------- government revenue ----------
    T_K_inv0    = τ_K_inv0*((1+n)*K1 - (1-δ_k)*K0)
    T_K0        = τ_K0*R_K_gross0*K0
    T_Ss_inv0   = (τ_S_inv0*((1+n)*S1 - (1-δ_s)*S0) 
                 + τ_s_inv0*((1+n)*s1 - (1-δ_s)*s0))
    T_L0        = ((τ_L_surf0 + τ_L_val0*p_LDV0)*L0 
                    + (τ_l_surf0 + τ_l_val0*p_LDV0)*l0
                    + τ_LDV_sale0*p_LR0*((1+n)*LDV1 - LDV0))
    T_L_R0      = ϱ*(τ_LR_surf0 + τ_LR_val0*p_LR0)*LR1 + τ_LR_sale0*p_LR0*((1+n)*LDV1 - LDV0)
    T_H0        = τ_H0*R_H_gross0*h0 + τ_HI0*R_H_gross0*H0
    
    # ℵitance tax
    T_ℵ0 = (
        ℵ*τ_ℵ_K0*K1 + 
        ℵ*τ_ℵ_S0*S1 + 
        ℵ*τ_ℵ_s0*s1 + 
        ℵ*τ_ℵ_H0*H0*R_H_gross0/ρ_H +
        ℵ*τ_ℵ_h0*h0*R_H_net0/ρ_h +
        ℵ*τ_ℵ_L0*p_LDV0*L1 + 
        ℵ*τ_ℵ_l0*p_LDV0*l1
        )
    T_ℵ_R0 = ϱ*ℵ*τ_ℵ_LR0*p_LR0*LR1

    # donation tax
    T_D0 = (
        d_K0*τ_D_K0*K1 + 
        d_S0*τ_D_S0*S1 + 
        d_s0*τ_D_s0*s1 + 
        d_H0*τ_D_H0*H0*R_H_gross0/ρ_H +
        d_h0*τ_D_h0*h0*R_H_net0/ρ_h +
        d_L0*τ_D_L0*p_LDV0*L1 + 
        d_l0*τ_D_l0*p_LDV0*l1
        )
    T_D_R0 = ϱ*d_LR0*τ_D_LR0*p_LR0*LR1
    
    T_total_C0 = T_K_inv0 + T_K0 + T_L0 + T_Ss_inv0 + T_H0 + T_ℵ0 + T_D0
    T_total_R0 = T_L_R0 + T_ℵ_R0 + T_D_R0
    T_total0   = T_total_C0 + T_total_R0

    
    
    # ---------------- Equilibrium conditions -----------------

    # FOCs and BCs
    # Euler on L
    loss[nvars*(t-1)+1] = (marg_benef_H*dH_dL1 - LM_C0*tax_wedge_L - LM_L0) # ^2
    
    # Euler on l
    loss[nvars*(t-1)+2] = (marg_benef_h*dh_dl1 - LM_C0*tax_wedge_l - LM_L0) # ^2
    
    # Euler on S
    loss[nvars*(t-1)+3] = (marg_benef_H*dH_dS1 + β*LM_C1*(1-δ_s)*((1+τ_S_inv1) + marg_adj_cost_S1) - LM_C0*tax_wedge_S) # ^2
    
    # Euler on s
    loss[nvars*(t-1)+4] = (marg_benef_h*dh_ds1 + β*LM_C1*(1-δ_s)*((1+τ_s_inv1) + marg_adj_cost_s1) - LM_C0*tax_wedge_s) # ^2
    
    # Euler on K
    loss[nvars*(t-1)+5] = (β*LM_C1*(R_K_net1 + (1-δ_k)*τ_K_inv1 + (1-δ_k)*marg_adj_cost_K1) - LM_C0*tax_wedge_K) # ^2
    
    # Euler on LR
    loss[nvars*(t-1)+6] = (β*R_R*LM_R_C1 - tax_wedge_LR*LM_R_C0 - LM_R_L0) # ^2
    
    # capitalist's BC
    loss[nvars*(t-1)+7] = (Y0 - ω*wage0
              + (1-δ_k)*K0 - (1+n)*K1        # capital
              + (1-δ_s)*S0 - (1+n)*S1        # S
              + (1-δ_s)*s0 - (1+n)*s1        # s
              + R_H_gross0*h0
              - C0 - ((1+n)*LDV1 -LDV0)*p_LR0 - total_cost
              - T_total_C0
              + T_total0/(1+ω)              # transfer to capitalists since they also work
              - adj_cost_K0 - adj_cost_S0 - adj_cost_s0    # adjustment cost
    ) # ^2

    # rentier's BC
    loss[nvars*(t-1)+8] = (R_R*LR0 + ((1+n)*LDV1 - LDV0)*p_LR0/ϱ
              - C_R0
              - T_total_R0/ϱ) # ^2
    
    # worker's BC
    loss[nvars*(t-1)+9] = (wage0 + T_total0/(1 + ω) - c0 - h0*R_H_gross0/ω) # ^2
    
end




function unpack_sol!(sol, Res, init_cond, terminal_cond, τ_path, params, T0)

    T_start = T0 + 1
    T_loc = T - T0

    (; ℵ, n, ω, ϱ, σ, β, γ_C, γ_R, A, δ_k, δ_s,
    α_C, α_K, a_L, cost_L1, cost_L2, Lbar, ρ_h, ρ_H, R_R, nvars) = params

    # recover transition sequences
    sol_path = [init_cond[10:end]; sol; terminal_cond[1:end-9]]

    K_path = @MVector zeros(T_loc+2)
    S_path = @MVector zeros(T_loc+2)
    s_path = @MVector zeros(T_loc+2)
    L_path = @MVector zeros(T_loc+2)
    l_path = @MVector zeros(T_loc+2)
    c_path = @MVector zeros(T_loc+2)
    C_path = @MVector zeros(T_loc+2)
    C_R_path  = @MVector zeros(T_loc+2)
    p_LR_path = @MVector zeros(T_loc+2)

    for t in 1:(T_loc+2)
        K_path[t] = sol_path[nvars*(t-1) + 1]
        S_path[t] = sol_path[nvars*(t-1) + 2]
        s_path[t] = sol_path[nvars*(t-1) + 3]
        L_path[t] = sol_path[nvars*(t-1) + 4]
        l_path[t] = sol_path[nvars*(t-1) + 5]

        c_path[t] = sol_path[nvars*(t-1) + 6]
        C_path[t] = sol_path[nvars*(t-1) + 7]
        C_R_path[t]  = sol_path[nvars*(t-1) + 8]
        p_LR_path[t] = sol_path[nvars*(t-1) + 9]
    end

    K_next_path = [K_path[2:end]; terminal_cond[5]]
    S_next_path = [S_path[2:end]; terminal_cond[6]]
    s_next_path = [s_path[2:end]; terminal_cond[7]]

    L_next_path = [L_path[2:end]; terminal_cond[8]]
    l_next_path = [l_path[2:end]; terminal_cond[9]]

    LDV_path =         @. L_path + l_path
    LDV_next_path =    @. L_next_path + l_next_path
    LR_next_path =     @. (Lbar - LDV_path) / ϱ
        
    # other derived variables
    k_path   = @. K_path/(1 + ω)
    LR_path  = @. (Lbar - (L_path + l_path)) / ϱ
    h_path   = @. h_prod(a_L, l_path, s_path)
    H_path   = @. h_prod(a_L, L_path, S_path)
    Y_path   = @. (1 + ω) * prod(α_K, A, k_path)
    p_LDV_path       = @. (1 + τ_path[:, nτ_LDV_sale]) * p_LR_path + cost_L1 + cost_L2*n*LDV_path
    R_K_gross_path   = @. dK_prod(α_K, A, k_path) + 1 - δ_k
    R_H_gross_path   = @. du_dh(c_path, h_path/ω, α_C, σ)/du_dc(c_path, h_path/ω, α_C, σ)

    wage_path    = @. (1 - α_K) * prod(α_K, A, k_path)
    u_path       = @. u(c_path, h_path/ω, α_C, σ)
    U_path       = @. u(C_path, H_path, α_C, σ)
    uR_path      = @. uR(C_R_path, σ)
    welfare_path = @. ω * u_path + γ_C * U_path + ϱ * γ_R * uR_path

    GDP_path = @. Y_path + R_H_gross_path*(H_path+h_path) + ϱ*R_R*LR_path

    marg_cost_path = @. cost_L1 + cost_L2*((1+n)*LDV_next_path - LDV_path)
    total_cost_path = @. cost_L1 * ((1+n)*LDV_next_path - LDV_path) + 1/2 * cost_L2 * ((1+n)*LDV_next_path - LDV_path)^2

    # unpack tax sequences
    τ_s_inv_path    = @. τ_path[:, nτ_s_inv]                 # capital, structure taxes
    τ_S_inv_path    = @. τ_path[:, nτ_S_inv]     
    τ_K_inv_path    = @. τ_path[:, nτ_K_inv]
    τ_K_path        = @. τ_path[:, nτ_K]
 
    τ_l_surf_path   = @. τ_path[:, nτ_l_surf]                 # land taxes 
    τ_l_val_path    = @. τ_path[:, nτ_l_val]
    τ_L_surf_path   = @. τ_path[:, nτ_L_surf]
    τ_L_val_path    = @. τ_path[:, nτ_L_val]
    τ_LR_surf_path  = @. τ_path[:, nτ_LR_surf]
    τ_LR_val_path   = @. τ_path[:, nτ_LR_val]
    τ_LDV_sale_path = @. τ_path[:, nτ_LDV_sale]
    τ_LR_sale_path  = @. τ_path[:, nτ_LR_sale]
    
    τ_H_path    = @. τ_path[:, nτ_H]
    τ_HI_path   = @. τ_path[:, nτ_HI]                     # housing taxes
 
    τ_D_K_path  = @. τ_path[:, nτ_D_K]                     # donation taxes
    τ_D_S_path  = @. τ_path[:, nτ_D_S]
    τ_D_s_path  = @. τ_path[:, nτ_D_s]
    τ_D_H_path  = @. τ_path[:, nτ_D_H]
    τ_D_h_path  = @. τ_path[:, nτ_D_h]
    τ_D_L_path  = @. τ_path[:, nτ_D_L]
    τ_D_l_path  = @. τ_path[:, nτ_D_l]
    τ_D_LR_path = @. τ_path[:, nτ_D_LR]
 
    τ_ℵ_K_path  = @. τ_path[:, nτ_ℵ_K]              # ℵitance taxes
    τ_ℵ_S_path  = @. τ_path[:, nτ_ℵ_S]
    τ_ℵ_s_path  = @. τ_path[:, nτ_ℵ_s]
    τ_ℵ_H_path  = @. τ_path[:, nτ_ℵ_H]
    τ_ℵ_h_path  = @. τ_path[:, nτ_ℵ_h]
    τ_ℵ_L_path  = @. τ_path[:, nτ_ℵ_L]
    τ_ℵ_l_path  = @. τ_path[:, nτ_ℵ_l]
    τ_ℵ_LR_path = @. τ_path[:, nτ_ℵ_LR]

    d_K_path     = n .+ ℵ.*τ_ℵ_K_path             # dilution
    d_H_path     = n .+ ℵ.*τ_ℵ_H_path
    d_h_path     = n .+ ℵ.*τ_ℵ_h_path
    d_S_path     = n .+ ℵ.*τ_ℵ_S_path
    d_s_path     = n .+ ℵ.*τ_ℵ_s_path
    d_L_path     = n .+ ℵ.*τ_ℵ_L_path
    d_l_path     = n .+ ℵ.*τ_ℵ_l_path
    d_LR_path    = n .+ ℵ.*τ_ℵ_LR_path

    R_H_net_path = @. (1-τ_H_path)*R_H_gross_path

    # tax revenue
    T_K_inv_path    = @. τ_K_inv_path*((1+n)*K_next_path - (1-δ_k)*K_path)
    T_K_path        = @. τ_K_path*R_K_gross_path
    T_Ss_inv_path   = @. τ_S_inv_path*((1+n)*S_next_path - (1-δ_s)*S_path) + τ_s_inv_path*((1+n)*s_next_path - (1-δ_s)*s_path)
    T_L_path        = @. ((τ_L_surf_path + τ_L_val_path*p_LDV_path)*L_path 
                            + (τ_l_surf_path + τ_l_val_path*p_LDV_path)*l_path
                            + τ_LDV_sale_path*p_LR_path*((1+n)*LDV_next_path - LDV_path))
    T_L_R_path      = @. ϱ*(τ_LR_surf_path + τ_LR_val_path*p_LR_path)*LR_next_path + τ_LR_sale_path*p_LR_path*((1+n)*LDV_next_path - LDV_path)
    T_H_path        = @. τ_H_path*R_H_gross_path*h_path + τ_HI_path*R_H_gross_path*H_path

    # ℵitance tax
    T_ℵ_path = @. (
            ℵ*τ_ℵ_K_path*K_next_path + 
            ℵ*τ_ℵ_S_path*S_next_path + 
            ℵ*τ_ℵ_s_path*s_next_path + 
            ℵ*τ_ℵ_H_path*H_path*R_H_gross_path/ρ_H +
            ℵ*τ_ℵ_h_path*h_path*R_H_net_path/ρ_h +
            ℵ*τ_ℵ_L_path*p_LDV_path*L_next_path + 
            ℵ*τ_ℵ_l_path*p_LDV_path*l_next_path
            )
    T_ℵ_R_path = @. ϱ*ℵ*τ_ℵ_LR_path*p_LR_path*LR_next_path

    # donation tax
    T_D_path = @. (
            d_K_path*τ_D_K_path*K_next_path + 
            d_S_path*τ_D_S_path*S_next_path + 
            d_s_path*τ_D_s_path*s_next_path + 
            d_H_path*τ_D_H_path*H_path*R_H_gross_path/ρ_H +
            d_h_path*τ_D_h_path*h_path*R_H_net_path/ρ_h +
            d_L_path*τ_D_L_path*p_LDV_path*L_next_path + 
            d_l_path*τ_D_l_path*p_LDV_path*l_next_path
            )
    T_D_R_path = @. ϱ*d_LR_path*τ_D_LR_path*p_LR_path*LR_next_path
    
    T_total_C_path =    @. T_K_inv_path + T_K_path + T_L_path + T_Ss_inv_path + T_H_path + T_ℵ_path + T_D_path
    T_total_R_path =    @. T_L_R_path + T_ℵ_R_path + T_D_R_path
    T_total_path =      @. T_total_C_path + T_total_R_path
    

    
    # -------- government revenue ----------
    T_K_inv_path    = @. τ_K_inv_path*((1+n)*K_next_path - (1-δ_k)*K_path)
    T_K_path        = @. τ_K_path*R_K_gross_path
    T_Ss_inv_path   = @. (τ_S_inv_path*((1+n)*S_next_path - (1-δ_s)*S_path) 
                        + τ_s_inv_path*((1+n)*s_next_path - (1-δ_s)*s_path))
    T_L_path        = @. ((τ_L_surf_path + τ_L_val_path*p_LDV_path)*L_path 
                        + (τ_l_surf_path + τ_l_val_path*p_LDV_path)*l_path
                        + τ_LDV_sale_path*p_LR_path*((1+n)*LDV_next_path - LDV_path))
    T_L_R_path      = @. ϱ*(τ_LR_surf_path + τ_LR_val_path*p_LR_path)*LR_next_path + τ_LR_sale_path*p_LR_path*((1+n)*LDV_next_path - LDV_path)
    T_H_path        = @. τ_H_path*R_H_gross_path*h_path + τ_HI_path*R_H_gross_path*H_path
    
    # ℵitance tax
    T_ℵ_path = @. (
            ℵ*τ_ℵ_K_path*K_next_path + 
            ℵ*τ_ℵ_S_path*S_next_path + 
            ℵ*τ_ℵ_s_path*s_next_path + 
            ℵ*τ_ℵ_H_path*H_path*R_H_gross_path/ρ_H +
            ℵ*τ_ℵ_h_path*h_path*R_H_net_path/ρ_h +
            ℵ*τ_ℵ_L_path*p_LDV_path*L_next_path + 
            ℵ*τ_ℵ_l_path*p_LDV_path*l_next_path
            )
    T_ℵ_R_path = @. ϱ*ℵ*τ_ℵ_LR_path*p_LR_path*LR_next_path

    # donation tax
    T_D_path = @. (
            d_K_path*τ_D_K_path*K_next_path + 
            d_S_path*τ_D_S_path*S_next_path + 
            d_s_path*τ_D_s_path*s_next_path + 
            d_H_path*τ_D_H_path*H_path*R_H_gross_path/ρ_H +
            d_h_path*τ_D_h_path*h_path*R_H_net_path/ρ_h +
            d_L_path*τ_D_L_path*p_LDV_path*L_next_path + 
            d_l_path*τ_D_l_path*p_LDV_path*l_next_path
            )
    T_D_R_path = @. ϱ*d_LR_path*τ_D_LR_path*p_LR_path*LR_next_path
    
    T_total_C_path = @. T_K_inv_path + T_K_path + T_L_path + T_Ss_inv_path + T_H_path + T_ℵ_path + T_D_path
    T_total_R_path = @. T_L_R_path + T_ℵ_R_path + T_D_R_path
    T_total_path   = @. T_total_C_path + T_total_R_path

    # save
    Res[T_start:end, 1]  = K_path
    Res[T_start:end, 2]  = S_path
    Res[T_start:end, 3]  = s_path
    Res[T_start:end, 4]  = L_path
    Res[T_start:end, 5]  = l_path
    Res[T_start:end, 6]  = c_path
    Res[T_start:end, 7]  = C_path
    Res[T_start:end, 8]  = C_R_path
    Res[T_start:end, 9]  = p_LR_path
    Res[T_start:end, 10] = K_next_path
    Res[T_start:end, 11] = S_next_path
    Res[T_start:end, 12] = s_next_path
    Res[T_start:end, 13] = L_next_path
    Res[T_start:end, 14] = l_next_path
    Res[T_start:end, 15] = LDV_next_path
    Res[T_start:end, 16] = LR_next_path
    Res[T_start:end, 17] = LDV_path
    Res[T_start:end, 18] = LR_path
    Res[T_start:end, 19] = h_path
    Res[T_start:end, 20] = H_path
    Res[T_start:end, 21] = Y_path
    Res[T_start:end, 22] = p_LDV_path
    Res[T_start:end, 23] = R_K_gross_path
    Res[T_start:end, 24] = R_H_gross_path
    Res[T_start:end, 25] = wage_path
    Res[T_start:end, 26] = u_path
    Res[T_start:end, 27] = U_path
    Res[T_start:end, 28] = uR_path
    Res[T_start:end, 29] = welfare_path
    Res[T_start:end, 30] = GDP_path
    Res[T_start:end, 31] = T_total_path
    Res[T_start:end, 32] = marg_cost_path
    Res[T_start:end, 33] = total_cost_path
    
end


########################################################
########################################################
########################################################



# -------------- define parameters --------------
# params = (; 
#     # population
#     ℵ = 0.01,
#     n = 0.01,
#     ω = 0.6956521739130435,
#     ϱ = 0.043478260869565216,

#     # preference
#     σ = 2.5,
#     β = 0.95,
#     γ_C = 0.25,
#     γ_R = 0.25,

#     # production
#     A   = 1,
#     δ_k = 0.09,
#     δ_s = 0.015,
#     α_C = 0.75,
#     α_K = 0.3333333333333,
#     a_L = 0.5,

#     # land
#     cost_L1 = 0.5,
#     cost_L2 = (0.62905859976161 - 0.5) / (1.36 * 0.01), # 46.25, # 0.62905859976161,
#     Lbar   = 2,
#     ρ_h = 0.2,
#     ρ_H = 0.2,
#     R_R = 0.178,
    
#     # size
#     nvars = 9
# )

# Read parameters from JSON file
param_data = JSON.parsefile("ParamDynamics.json")

params = (
    # population
    ℵ = param_data["aleph"],
    n = param_data["n"],
    ω = param_data["omega"],
    ϱ = param_data["varrho"],

    # preference
    σ = param_data["sigma"],
    β = param_data["beta"],
    γ_C = param_data["gamma_C"],
    γ_R = param_data["gamma_R"],

    # production
    A   = param_data["A"],
    δ_k = param_data["delta_K"],
    δ_s = param_data["delta_S"],
    α_C = param_data["alpha_CH"],
    α_K = param_data["alpha_K"],
    a_L = param_data["a_L"],

    # land
    cost_L1 = param_data["cost_L1"],
    cost_L2 = param_data["cost_L2"],
    Lbar   = param_data["Lbar"],
    ρ_h = param_data["rho_h"],
    ρ_H = param_data["rho_H"],
    R_R = param_data["R_R"],
    
    # size
    nvars = 9
)



# ---------------- solve dynamics ---------------

################## STEP 0: Pre-define parameters ##################
T = 600
T1 = 100            # t = 0 is time of annoucement, t = T1 is time of implement

# all taxes to 0
τ_path_PostT1_eq0 = zeros(T-T1+2, 30)
τ_path_eq0 = zeros(T+2, 30)


# tax index
const nτ_s_inv    = 1                 # capital, structure taxes
const nτ_S_inv    = 2     
const nτ_K_inv    = 3
const nτ_K        = 4

const nτ_l_surf   = 5                # land taxes 
const nτ_l_val    = 6
const nτ_L_surf   = 7
const nτ_L_val    = 8
const nτ_LR_surf  = 9
const nτ_LR_val   = 10
const nτ_LDV_sale = 11
const nτ_LR_sale  = 12

const nτ_H    = 13
const nτ_HI   = 14                     # housing taxes

const nτ_D_K  = 15                     # donation taxes
const nτ_D_S  = 16
const nτ_D_s  = 17
const nτ_D_H  = 18
const nτ_D_h  = 19
const nτ_D_L  = 20
const nτ_D_l  = 21
const nτ_D_LR = 22

const nτ_ℵ_K  = 23                     # ℵitance taxes
const nτ_ℵ_S  = 24
const nτ_ℵ_s  = 25
const nτ_ℵ_H  = 26
const nτ_ℵ_h  = 27
const nτ_ℵ_L  = 28
const nτ_ℵ_l  = 29
const nτ_ℵ_LR = 30


# -------------------------------------------------------------
# ---------- create experiments we want to run ----------------
# -------------------------------------------------------------


# Experiment 1: A negative shock in initial capital stock
τ_path_PostT1_exprmt1   = SMatrix{T-T1+2, 30}(τ_path_PostT1_eq0)
τ_path_exprmt1          = SMatrix{T+2, 30}(τ_path_eq0)



# Experiment 2: A suddent jump in tax on capital
τ_path_PostT1_exprmt2   = copy(τ_path_PostT1_eq0)
τ_path_exprmt2          = copy(τ_path_eq0)

τ_path_PostT1_exprmt2[:, nτ_K]  .= 0.011
τ_path_exprmt2[T1+1:end, :]         = τ_path_PostT1_exprmt2                     # MIT shock sequence

τ_path_PostT1_exprmt2   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt2)         
τ_path_exprmt2          = SMatrix{T+2, 30}(τ_path_exprmt2)



# Experiment 3: A sudden jump in uniform tax on land surface
τ_path_PostT1_exprmt3   = copy(τ_path_PostT1_eq0)
τ_path_exprmt3          = copy(τ_path_eq0)

τ_path_PostT1_exprmt3[:, nτ_L_surf]  .= 0.02
τ_path_PostT1_exprmt3[:, nτ_l_surf]  .= 0.02
τ_path_PostT1_exprmt3[:, nτ_LR_surf]  .= 0.02
τ_path_exprmt3[T1+1:end, :]         = τ_path_PostT1_exprmt3                     # MIT shock sequence

τ_path_PostT1_exprmt3   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt3)         
τ_path_exprmt3          = SMatrix{T+2, 30}(τ_path_exprmt3)



# Experiment 4: Gradual increase in uniform tax on land surface to 2%
τ_path_PostT1_exprmt4   = copy(τ_path_PostT1_eq0)
τ_path_exprmt4          = copy(τ_path_eq0)

ϕ = 0.95         # smoothing parameter
prog = reverse([ϕ^t * 0.02 for t in 1:100])

τ_path_PostT1_exprmt4[:, nτ_L_surf]  = [prog; 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt4[:, nτ_l_surf]  = [prog; 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt4[:, nτ_LR_surf] = [prog; 0.02 * ones(T-T1+2 - 100)]
τ_path_exprmt4[T1+1:end, :]         = τ_path_PostT1_exprmt4                     # MIT shock sequence

τ_path_PostT1_exprmt4   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt4)         
τ_path_exprmt4          = SMatrix{T+2, 30}(τ_path_exprmt4)



# Experiment 5: Linear increase in uniform tax on land surface to 2%
τ_path_PostT1_exprmt5   = copy(τ_path_PostT1_eq0)
τ_path_exprmt5          = copy(τ_path_eq0)

τ_path_PostT1_exprmt5[:, nτ_L_surf]  = [LinRange(0.0, 0.02, 100); 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt5[:, nτ_l_surf]  = [LinRange(0.0, 0.02, 100); 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt5[:, nτ_LR_surf] = [LinRange(0.0, 0.02, 100); 0.02 * ones(T-T1+2 - 100)]
τ_path_exprmt5[T1+1:end, :]         = τ_path_PostT1_exprmt5                     # MIT shock sequence

τ_path_PostT1_exprmt5   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt5)         
τ_path_exprmt5          = SMatrix{T+2, 30}(τ_path_exprmt5)



# Experiment 6: Smoothed linear increase in uniform tax on land surface to 2%
τ_path_PostT1_exprmt6   = copy(τ_path_PostT1_eq0)
τ_path_exprmt6          = copy(τ_path_eq0)

g(x; c, x1, x2, s) = (
    c/2 + (1/2)*(c/(x2 - x1))*((x-x1)^2 + (c/(x2 - x1))^2 * s^2)^(1/2) 
    - (1/2)*c/(x2 - x1)*((x-x2)^2 + (c/(x2 - x1))^2 * s^2)^(1/2)
)

smoothed_path = g.(1:T-T1+2; c=0.02, x1=50, x2=150, s=3e4)

τ_path_PostT1_exprmt6[:, nτ_L_surf]  = smoothed_path
τ_path_PostT1_exprmt6[:, nτ_l_surf]  = smoothed_path
τ_path_PostT1_exprmt6[:, nτ_LR_surf] = smoothed_path
τ_path_exprmt6[T1+1:end, :]         = τ_path_PostT1_exprmt6                     # MIT shock sequence

τ_path_PostT1_exprmt6   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt6)         
τ_path_exprmt6          = SMatrix{T+2, 30}(τ_path_exprmt6)



# Experiment 7: A sudden jump in uniform (rate) tax on land value
τ_path_PostT1_exprmt7   = copy(τ_path_PostT1_eq0)
τ_path_exprmt7          = copy(τ_path_eq0)

τ_path_PostT1_exprmt7[:, nτ_L_val]  .= 0.02
τ_path_PostT1_exprmt7[:, nτ_l_val]  .= 0.02
τ_path_PostT1_exprmt7[:, nτ_LR_val]  .= 0.02
τ_path_exprmt7[T1+1:end, :]         = τ_path_PostT1_exprmt7                     # MIT shock sequence

τ_path_PostT1_exprmt7   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt7)         
τ_path_exprmt7          = SMatrix{T+2, 30}(τ_path_exprmt7)



# Experiment 8: Gradual increase in uniform (rate) tax on land value to 2%
τ_path_PostT1_exprmt8   = copy(τ_path_PostT1_eq0)
τ_path_exprmt8          = copy(τ_path_eq0)

ϕ = 0.95         # smoothing parameter
prog = reverse([ϕ^t * 0.02 for t in 1:100])

τ_path_PostT1_exprmt8[:, nτ_L_val]  = [prog; 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt8[:, nτ_l_val]  = [prog; 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt8[:, nτ_LR_val] = [prog; 0.02 * ones(T-T1+2 - 100)]
τ_path_exprmt8[T1+1:end, :]         = τ_path_PostT1_exprmt8                     # MIT shock sequence

τ_path_PostT1_exprmt8   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt8)         
τ_path_exprmt8          = SMatrix{T+2, 30}(τ_path_exprmt8)



# Experiment 9: Linear increase in uniform (rate) tax on land value to 2%
τ_path_PostT1_exprmt9   = copy(τ_path_PostT1_eq0)
τ_path_exprmt9          = copy(τ_path_eq0)

τ_path_PostT1_exprmt9[:, nτ_L_val]  = [LinRange(0.0, 0.02, 100); 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt9[:, nτ_l_val]  = [LinRange(0.0, 0.02, 100); 0.02 * ones(T-T1+2 - 100)]
τ_path_PostT1_exprmt9[:, nτ_LR_val] = [LinRange(0.0, 0.02, 100); 0.02 * ones(T-T1+2 - 100)]
τ_path_exprmt9[T1+1:end, :]         = τ_path_PostT1_exprmt9                     # MIT shock sequence

τ_path_PostT1_exprmt9   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt9)         
τ_path_exprmt9          = SMatrix{T+2, 30}(τ_path_exprmt9)



# Experiment 10: Smoothed linear increase in uniform (rate) tax on land value to 2%
τ_path_PostT1_exprmt10   = copy(τ_path_PostT1_eq0)
τ_path_exprmt10          = copy(τ_path_eq0)

g(x; c, x1, x2, s) = (
    c/2 + (1/2)*(c/(x2 - x1))*((x-x1)^2 + (c/(x2 - x1))^2 * s^2)^(1/2) 
    - (1/2)*c/(x2 - x1)*((x-x2)^2 + (c/(x2 - x1))^2 * s^2)^(1/2)
)

smoothed_path = g.(1:T-T1+2; c=0.02, x1=50, x2=150, s=3e4)

τ_path_PostT1_exprmt10[:, nτ_L_val]  = smoothed_path
τ_path_PostT1_exprmt10[:, nτ_l_val]  = smoothed_path
τ_path_PostT1_exprmt10[:, nτ_LR_val] = smoothed_path
τ_path_exprmt10[T1+1:end, :]         = τ_path_PostT1_exprmt10                     # MIT shock sequence

τ_path_PostT1_exprmt10   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt10)         
τ_path_exprmt10          = SMatrix{T+2, 30}(τ_path_exprmt10)




# Experiment 11: A sudden jump in tax on land transaction to 5%
τ_path_PostT1_exprmt11   = copy(τ_path_PostT1_eq0)
τ_path_exprmt11          = copy(τ_path_eq0)

τ_path_PostT1_exprmt11[:, nτ_LDV_sale]  .= 0.02
τ_path_exprmt11[T1+1:end, :]         = τ_path_PostT1_exprmt11                     # MIT shock sequence

τ_path_PostT1_exprmt11   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt11)         
τ_path_exprmt11          = SMatrix{T+2, 30}(τ_path_exprmt11)




# Experiment 12: A transient (one-period) increase in capital tax to 1.1%
τ_path_PostT1_exprmt12   = copy(τ_path_PostT1_eq0)
τ_path_exprmt12          = copy(τ_path_eq0)

τ_path_PostT1_exprmt12[:, nτ_K]  .= [0.011; zeros(T-T1+2-1)]
τ_path_exprmt12[T1+1:end, :]         = τ_path_PostT1_exprmt12                     # MIT shock sequence

τ_path_PostT1_exprmt12   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt12)         
τ_path_exprmt12          = SMatrix{T+2, 30}(τ_path_exprmt12)




# # Experiment 13: A permanent jump in tax on raw land surface to 50% at t=100
# τ_path_PostT1_exprmt13   = copy(τ_path_PostT1_eq0)
# τ_path_exprmt13          = copy(τ_path_eq0)

# τ_path_PostT1_exprmt13[:, nτ_LR_surf]  = 0.1 * ones(T-T1+2)
# τ_path_exprmt13[T1+1:end, :]         = τ_path_PostT1_exprmt13                     # MIT shock sequence

# τ_path_PostT1_exprmt13   = SMatrix{T-T1+2, 30}(τ_path_PostT1_exprmt13)         
# τ_path_exprmt13          = SMatrix{T+2, 30}(τ_path_exprmt13)


τ_path_exprmts = [
    (τ_path_PostT1_exprmt1, τ_path_exprmt1)
    (τ_path_PostT1_exprmt2, τ_path_exprmt2)
    (τ_path_PostT1_exprmt3, τ_path_exprmt3)
    (τ_path_PostT1_exprmt4, τ_path_exprmt4)
    (τ_path_PostT1_exprmt5, τ_path_exprmt5)
    (τ_path_PostT1_exprmt6, τ_path_exprmt6)
    (τ_path_PostT1_exprmt7, τ_path_exprmt7)
    (τ_path_PostT1_exprmt8, τ_path_exprmt8)
    (τ_path_PostT1_exprmt9, τ_path_exprmt9)
    (τ_path_PostT1_exprmt10, τ_path_exprmt10)
    (τ_path_PostT1_exprmt11, τ_path_exprmt11)
    (τ_path_PostT1_exprmt12, τ_path_exprmt12)
    # (τ_path_PostT1_exprmt13, τ_path_exprmt13)
    ]


for k in 1:12 # 12

    println("=============== Experiment $(k) ===============")

    # unpack τ_path_PostT1_exprmt
    (τ_path_PostT1, τ_path)  = τ_path_exprmts[k]


    ############# STEP 1: SET INITIAL AND TERMINAL CONDITIONS #############
    
    init_ss_js      = JSON.parsefile("dynamics/exprmt$(k)_init.json")
    terminal_ss_js  = JSON.parsefile("dynamics/exprmt$(k)_terminal.json")

    init_ss = [init_ss_js["K"],
               init_ss_js["S"],
               init_ss_js["s"],
               init_ss_js["L"],
               init_ss_js["l"],
               init_ss_js["c"],
               init_ss_js["C"],
               init_ss_js["C_R"],
               init_ss_js["p_LR"]]

    terminal_ss = [terminal_ss_js["K"],
                   terminal_ss_js["S"],
                   terminal_ss_js["s"],
                   terminal_ss_js["L"],
                   terminal_ss_js["l"],
                   terminal_ss_js["c"],
                   terminal_ss_js["C"],
                   terminal_ss_js["C_R"],
                   terminal_ss_js["p_LR"]]

    init_ss_derived = [
        init_ss_js["K"],    # next period
        init_ss_js["S"],     # next period
        init_ss_js["s"],     # next period
        init_ss_js["L"],     # next period
        init_ss_js["l"],     # next period
        init_ss_js["LDV"],   # next period
        init_ss_js["LR"],   # next period
        init_ss_js["LDV"],   # this period
        init_ss_js["LR"],    # this period
        init_ss_js["h"],
        init_ss_js["H"],
        init_ss_js["Y"],
        init_ss_js["p_LDV"],
        init_ss_js["R_K_gross"],
        init_ss_js["R_H_gross"],
        init_ss_js["wage"],
        init_ss_js["util_c"],
        init_ss_js["util_C"],
        init_ss_js["util_CR"],
        init_ss_js["welfare"],
        init_ss_js["GDP"],
        init_ss_js["T_total"],
        init_ss_js["marginal_cost"],
        init_ss_js["total_cost"]
        ]
    
    # if k==1            # negative capital shock experiment - terminal_ss is the steady state before the shock
    #     init_ss0 = terminal_ss
    # else
    #     init_ss0 = init_ss
    # end
    init_ss0 = init_ss

    init_cond     = SVector{5 + 9}([init_ss0; init_ss[1:5]])
    terminal_cond = SVector{4 + 9 + 9}([terminal_ss[6:9]; terminal_ss; terminal_ss])

    # create result matrices
    Res_PF           = @MMatrix zeros(T+2, 9 + 23 + 1 + 30)
    Res_PF[:, 1:9]   = repeat(init_ss', T+2)
    Res_PF[:, 10:33] = repeat(init_ss_derived', T+2)
    Res_PF[T1+1:end, 34:end] = τ_path_PostT1

    Res_MIT  = copy(Res_PF)


    ################## STEP 2: solve MIT shock sequence ##################

    # set initial guess xx0
    xx0 = repeat(terminal_ss, T-T1-1)
    xx0 = [terminal_ss[6:9]; xx0; terminal_ss[1:5]] .* 0.9

    # initialize loss vector
    loss = zeros(9*(T-T1+1))

    # solve
    f2solve_MIT!(loss, xx0) = transition_loss!(loss, xx0, init_cond, terminal_cond, τ_path_PostT1, T-T1, params)

    @time res= optimize!(
        LeastSquaresProblem(x = xx0, f! = f2solve_MIT!, output_length = 9*(T-T1+1), autodiff = :forward), 
        Dogleg())

    sol = res.minimizer

    # report SSR
    println("MIT: SSR = ", res.ssr)

    # unpack and save
    unpack_sol!(sol, Res_MIT, init_cond, terminal_cond, τ_path_PostT1, params, T1)

    writedlm( "dynamics/exprmt$(k)_MIT.csv",  Res_MIT, ',')
    # writedlm( "dynamics/exprmt$(k)_MIT_loss.csv",  loss, ',')


    ################## STEP 3: solve Perfect Foresight sequence ##################

    # set initial guess xx0
    xx0 = repeat(terminal_ss, T-1)
    xx0 = [terminal_ss[6:9]; xx0; terminal_ss[1:5]] .* 0.9

    # initialize loss vector
    loss = zeros(9*(T+1))

    # solve
    f2solve_PF!(loss, xx0) = transition_loss!(loss, xx0, init_cond, terminal_cond, τ_path, T, params)

    @time res= optimize!(
        LeastSquaresProblem(x = xx0, f! = f2solve_PF!, output_length = 9*(T+1), autodiff = :forward), 
        Dogleg())

    sol = res.minimizer

    # report SSR
    println("PF: SSR = ", res.ssr)

    # unpack and save
    unpack_sol!(sol, Res_PF, init_cond, terminal_cond, τ_path, params, 0)

    writedlm( "dynamics/exprmt$(k)_PF.csv",  Res_PF, ',')
    # writedlm( "dynamics/exprmt$(k)_PF_loss.csv",  loss, ',')
    
    
end