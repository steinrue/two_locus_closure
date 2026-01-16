## Imports and path set-up

import os
import numpy as np
import matplotlib.pyplot as plt
import variables

## Population Size History Plot
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(4, 3))

# Offset constant and bottleneck population for visualization
ax.plot(range(len(variables.popScenarios['constant'])), variables.popScenarios['constant']-110, label='Constant', color=colors[0])
ax.plot(range(len(variables.popScenarios['bottleneck'])), variables.popScenarios['bottleneck']+40, label='Bottleneck', color=colors[1])
ax.plot(range(len(variables.popScenarios['expGrowth'])), variables.popScenarios['expGrowth'], label='Exp. Growth', color=colors[2])
ax.set_yscale('log')
ax.set_ylim(200, 50000)
ax.set_yticks([200, 500, 2000, 40000])
ax.minorticks_off()
ax.set_yticklabels([f'{variables.scientific_not(t)}' for t in [200, 500, 2000, 40000]])
ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Population Size', fontsize=12)
ax.legend()

fig.suptitle('Population Size Histories')
fig.tight_layout()
fig_filename = os.path.join(variables.figDir, 'populationSize.pdf')
fig.savefig(fig_filename, format='pdf', bbox_inches='tight')
plt.show()