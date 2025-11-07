import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

try:  # pragma: no cover - fallback for direct execution
    from .paths import DYNAMICS_DIR
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

    from land_tax.paths import DYNAMICS_DIR  # type: ignore


#######################################################
#################### Function #########################
#######################################################
def plot_dynamics(ax, exprmt, init_ss, tax_rate_id, lab, ls, col=None):
    
    ax[0,0].set_title(r'$K$')
    ax[0,0].plot((exprmt[:,0] - init_ss['K'])/np.abs(init_ss['K']), ls=ls, color=col)
    ax[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[0,1].set_title(r'$S$')
    ax[0,1].plot((exprmt[:,1] - init_ss['S'])/np.abs(init_ss['S']), ls=ls, color=col)
    ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[0,2].set_title(r'$s$')
    ax[0,2].plot((exprmt[:,2] - init_ss['s'])/np.abs(init_ss['s']), ls=ls, color=col)
    ax[0,2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[0,3].set_title(r'$Y$')
    ax[0,3].plot((exprmt[:,20] - init_ss['Y'])/init_ss['Y'], ls=ls, color=col)
    ax[0,3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    ax[0,4].set_title(r'$K_{inv}/K^{*}_{init}$')
    ax[0,4].plot((exprmt[:,9] - exprmt[:,0])/init_ss['K'], ls=ls, color=col)
    ax[0,4].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[1,0].set_title(r'$\mathcal{L}$')
    ax[1,0].plot((exprmt[:,3] - init_ss['L'])/init_ss['L'], ls=ls, color=col)
    ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[1,1].set_title(r'$\mathcal{l}$')
    ax[1,1].plot((exprmt[:,4] - init_ss['l'])/init_ss['l'], ls=ls, color=col)
    ax[1,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[1,2].set_title(r'$\mathcal{L}_{R}$')
    ax[1,2].plot((exprmt[:,17] - init_ss['LR'])/init_ss['LR'], ls=ls, color=col)
    ax[1,2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[1,3].set_title(r'$p^{\mathcal{L}_{R}}$')
    ax[1,3].plot((exprmt[:,8] - init_ss['p_LR'])/init_ss['p_LR'], ls=ls, color=col)
    ax[1,3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    ax[1,4].set_title(r'$S_{inv}/S^{*}_{init}$')
    ax[1,4].plot((exprmt[:,10] - exprmt[:,1])/init_ss['S'], ls=ls, color=col)
    ax[1,4].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[2,0].set_title(r'$H$')
    ax[2,0].plot((exprmt[:,19] - init_ss['H'])/init_ss['H'], ls=ls, color=col)
    ax[2,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[2,1].set_title(r'$h$')
    ax[2,1].plot((exprmt[:,18] - init_ss['h'])/init_ss['h'], ls=ls, color=col)
    ax[2,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[2,2].set_title(r'$R^{H}_{gross}$')
    ax[2,2].plot((exprmt[:,23] - init_ss['R_H_gross'])/init_ss['R_H_gross'], ls=ls, color=col)
    ax[2,2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[2,3].set_title(r'$R^{K}_{gross}$')
    ax[2,3].plot((exprmt[:,22] - init_ss['R_K_gross'])/init_ss['R_K_gross'], ls=ls, color=col)
    ax[2,3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    ax[2,4].set_title(r'$s_{inv}/s^{*}_{init}$')
    ax[2,4].plot((exprmt[:,11] - exprmt[:,2])/init_ss['s'], ls=ls, color=col)
    ax[2,4].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[3,0].set_title(r'$C$')
    ax[3,0].plot((exprmt[:,6] - init_ss['C'])/init_ss['C'], ls=ls, color=col)
    ax[3,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[3,1].set_title(r'$c$')
    ax[3,1].plot((exprmt[:,5] - init_ss['c'])/init_ss['c'], ls=ls, color=col)
    ax[3,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[3,2].set_title(r'$C_R$')
    ax[3,2].plot((exprmt[:,7] - init_ss['C_R'])/init_ss['C_R'], ls=ls, color=col)
    ax[3,2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[3,3].set_title('Wage')
    ax[3,3].plot((exprmt[:,24] - init_ss['wage'])/init_ss['wage'], ls=ls, color=col)
    ax[3,3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # new
    ax[3,4].set_title(r'$(LDV_{t+1}-LDV_{t})/LDV^{*}_{init}$')
    ax[3,4].plot((exprmt[:,14] - exprmt[:,16])/init_ss['LDV'], ls=ls, color=col)
    ax[3,4].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    ax[4,0].set_title('GDP')
    ax[4,0].plot((exprmt[:,29] - init_ss['GDP'])/init_ss['GDP'], ls=ls, color=col)
    ax[4,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[4,1].set_title('Tax rate')
    ax[4,1].plot(exprmt[:,tax_rate_id], ls=ls, color=col)

    transfer2GDP = exprmt[:,30]/init_ss['GDP']
    ax[4,2].set_title('Transfer/GDP')
    ax[4,2].plot(transfer2GDP, ls=ls, color=col)
    ax[4,2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[4,3].set_title('Welfare')
    ax[4,3].plot((exprmt[:,28] - init_ss['welfare'])/np.abs(init_ss['welfare']), ls=ls, label=lab, color=col)
    ax[4,3].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax[4,4].set_title('Marginal land development cost')
    ax[4,4].plot(exprmt[:,31], ls=ls, label=lab, color=col)
    # ax[4,4].plot((exprmt[:,31] - init_ss['marginal_cost'])/np.abs(init_ss['marginal_cost']), ls=ls, label=lab, color=col)
    # ax[4,4].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    for i in range(4):
        ax[i,i].set_xlim([-10,500])
        ax[4,i].set_xlabel(r'$t$')




#######################################################
#######################################################
#######################################################

# Experiment 1
exprmt1 = np.loadtxt(DYNAMICS_DIR / "exprmt1_PF.csv", delimiter=",")

with open(DYNAMICS_DIR / 'exprmt1_terminal.json') as file:
    init_ss_exprmt1 = json.load(file)       # terminal is the steady state capital stock (init is the shocked K)

fig, ax = plt.subplots(5, 5, figsize=[15, 9], dpi=200, sharex=True)

plot_dynamics(ax, exprmt1, init_ss_exprmt1, 31, None, '-')

fig.suptitle('A 20% negative shock to the capital stock')
# fig.delaxes(ax[-1,-1])
plt.tight_layout()

# plt.savefig('Figures/Dynamics_MIT/Dynamics_experiment1.png', bbox_inches='tight')
# plt.savefig('Figures/Dynamics_PF/Dynamics_experiment1.png', bbox_inches='tight')
# plt.savefig('Figures/Dynamics_both/Dynamics_experiment1.png', bbox_inches='tight')

plt.savefig('Figures/Dynamics_adjcost_MIT/Dynamics_experiment1.png', bbox_inches='tight')
plt.savefig('Figures/Dynamics_adjcost_PF/Dynamics_experiment1.png', bbox_inches='tight')
plt.savefig('Figures/Dynamics_adjcost_both/Dynamics_experiment1.png', bbox_inches='tight')

plt.close()



#######################################################

Titles = [
    r'A permanent increase in capital tax to 1.1% at $t=100$',
    r'A permanent increase in uniform tax on land surface to 2% at $t=100$',
    r'Geometric increase in uniform tax on land surface to 2% from $t=100$',
    r'Linear increase in uniform tax on land surface to 2% from $t=100$',
    r'Smoothed linear increase in uniform tax on land surface to 2% from $t=100$',
    r'A permanent increase in uniform-rate tax on land value to 2% at $t=100$',
    r'Geometric increase in uniform-rate tax on land value to 2% from $t=100$',
    r'Linear increase in uniform-rate tax on land value to 2% from $t=100$',
    r'Smoothed linear increase in uniform-rate tax on land value to 2% from $t=100$',
    r'A permanent increase in tax on land purchase to 2% at $t=100$',
    r'A transient (one-period) increase in capital tax to 1.1% at $t=100$',
    # r'Experiment 13: Permanent jump in tax on raw land surface to 50% at $t=100$'
]

tax_ids = [
    31 + 3,
    31 + 4,
    31 + 4,
    31 + 4,
    31 + 4,
    31 + 5,
    31 + 5,
    31 + 5,
    31 + 5,
    31 + 10,
    31 + 3,
    # 31 + 8,
]

for i in range(2, 13):
    exprmt_PF = np.loadtxt(DYNAMICS_DIR / f"exprmt{i}_PF.csv", delimiter=",")
    exprmt_MIT = np.loadtxt(DYNAMICS_DIR / f"exprmt{i}_MIT.csv", delimiter=",")
    
    with open(DYNAMICS_DIR / f'exprmt{i}_init.json') as file:
        init_ss_exprmt = json.load(file)

    tax_id = tax_ids[i-2]

    ######
    fig, ax = plt.subplots(5, 5, figsize=[15, 9], dpi=200, sharex=True)

    plot_dynamics(ax, exprmt_MIT, init_ss_exprmt, tax_id, 'As a surprise', '-', 'tab:blue')       # 'As a surprise', 
    fig.suptitle(Titles[i-2])
    # fig.delaxes(ax[-1,-1])
    plt.tight_layout()
    plt.savefig(f'Figures/Dynamics_adjcost_MIT/Dynamics_experiment{i}.png', bbox_inches='tight')
    plt.close()

    ######
    fig, ax = plt.subplots(5, 5, figsize=[15, 9], dpi=200, sharex=True)

    plot_dynamics(ax, exprmt_PF, init_ss_exprmt, tax_id, r'Pre-announced at $t=0$', '--', 'tab:orange')        # r'Pre-announced at $t=0$',
    fig.suptitle(Titles[i-2])
    # fig.delaxes(ax[-1,-1])
    plt.tight_layout()
    plt.savefig(f'Figures/Dynamics_adjcost_PF/Dynamics_experiment{i}.png', bbox_inches='tight')
    plt.close()


    ######
    fig, ax = plt.subplots(5, 5, figsize=[15, 9], dpi=200, sharex=True)

    plot_dynamics(ax, exprmt_PF, init_ss_exprmt, tax_id, r'Pre-announced at $t=0$', '--', 'tab:orange')
    plot_dynamics(ax, exprmt_MIT, init_ss_exprmt, tax_id, 'As a surprise', '-', 'tab:blue')
    fig.suptitle(Titles[i-2])
    # fig.delaxes(ax[-1,-1])
    plt.tight_layout()
    ax[4,3].legend(bbox_to_anchor=(0, -0.4), ncol=2)
    plt.savefig(f'Figures/Dynamics_adjcost_both/Dynamics_experiment{i}.png', bbox_inches='tight')
    plt.close()