from utils import *

spike_pad = 2
swim_pad = 30
visu_pad = 60
_ = np.load('../KernelFits/depreciated/subvolt_fit.npz', allow_pickle=True)

print('Cell selectivity')
comp = pd.DataFrame(_['comp_list'], columns=['swim vigor', 'visual input'])
comp = comp.replace(-1, 'inh')
comp = comp.replace(1, 'exc')
comp_table=pd.crosstab(comp['swim vigor'], comp['visual input'], margins = False)

plt.figure()
sns.heatmap(comp_table, annot=True, fmt='d')
plt.yticks([0.5,1.5])
plt.xticks([0.5,1.5])
plt.ylim([0, 2])
plt.xlim([0, 2])
plt.title('Number of cells')
plt.savefig('../Plots/gain/kernel_subvolt_cell_sel.pdf')
plt.close('all')

print('Explained variance')
comp = _['comp_list']
comp_ind = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
comp_title = ['-/-', '-/+', '+/-', '+/+']
ev_model = _['ev_model_list']*100
ev_spike = _['ev_spike_list']*100
ev_swim = _['ev_swim_list']*100
ev_visu = _['ev_visual_list']*100
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax = ax.flatten()
for n_ in range(4):
    comp_ = comp_ind[n_]
    valid_ind = (comp==comp_).sum(axis=-1)==2
    ax[n_].plot(['spike', 'swim', 'visual'],[ev_spike[valid_ind], ev_swim[valid_ind], ev_visu[valid_ind]], '-ok')
    ax[n_].set_title(comp_title[n_])
    ax[n_].set_ylabel('EV (%)')
    sns.despine()
plt.tight_layout()
plt.savefig('../Plots/gain/kernel_subvolt_ev.pdf')
plt.close('all')

print('Swim kernel')
w_ = _['w_list']
comp_ = comp_ind[1]
valid_ind = (comp==comp_).sum(axis=-1)==2
w_swim = w_[valid_ind, (spike_pad+4):(spike_pad+swim_pad*2-4)]
w_swim_norm = w_swim/w_swim.max(axis=-1, keepdims=True)
w_ind = np.argmax(w_swim_norm, axis=-1)
w_order = np.argsort(w_ind)
plt.figure()
plt.imshow(w_swim_norm[w_order], extent=(-swim_pad/300, swim_pad/300, 1, len(w_order)), aspect='auto')
plt.yticks([1, len(w_order)])
plt.ylabel('Neuron index')
plt.xlabel('Kernel time (s)')
plt.title('Swim kernel')
plt.savefig('../Plots/gain/kernel_subvolt_swim_weight.pdf')
plt.close('all')

print('Visual kernel')
w_ = _['w_list']
comp_ = comp_ind[1]
valid_ind = (comp==comp_).sum(axis=-1)==2
w_visu = w_[valid_ind, (spike_pad+swim_pad*2+4):-4]
w_visu_norm = w_visu/(w_visu.max(axis=-1, keepdims=True)+0.000001)
w_ind = np.argmax(w_visu_norm, axis=-1)
w_order = np.argsort(w_ind)
plt.figure()
plt.imshow(w_visu_norm[w_order], extent=(-visu_pad/300, 0, 1, len(w_order)), aspect='auto')
plt.yticks([1, len(w_order)])
plt.ylabel('Neuron index')
plt.xlabel('Kernel time (s)')
plt.title('visual kernel')
plt.savefig('../Plots/gain/kernel_subvolt_visual_weight.pdf')
plt.close('all')