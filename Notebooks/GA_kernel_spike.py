from utils import *

t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_label = np.arange(-t_pre, t_post)/300
t_sig = 240
sub_pad=9
visu_pad=200

_ = np.load('../KernelFits/depreciated/spike_fit.npz', allow_pickle=True)
ev_model = _['ev_full_list']*100
ev_sub = _['ev_sub_list']*100
explain_ratio = ev_sub/(ev_model-ev_sub)
plt.figure()
plt.hist(np.log(explain_ratio))
plt.ylabel('# Cells')
plt.xlabel('Log $EV_{visu}/EV_{sub}$')
plt.savefig('../Plots/gain/kernel_spike_ev.pdf')
plt.close('all')

explain_ratio = ev_sub/ev_model
w_ = _['wfull_list'].squeeze()
valid_ind = explain_ratio<0.7
w_swim = w_[valid_ind, (sub_pad+1):]
w_swim_norm = w_swim/np.abs(w_swim).max(axis=-1, keepdims=True)
w_ind = np.argmax(w_swim_norm, axis=-1)
w_order = np.argsort(w_ind)
plt.figure()
plt.imshow(w_swim_norm[w_order], extent=(-visu_pad/300, 0/300, 1, len(w_order)), aspect='auto')
plt.yticks([1, len(w_order)])
plt.ylabel('Neuron index')
plt.xlabel('Kernel time (s)')
plt.title('Visual kernel')
plt.colorbar()
plt.savefig('../Plots/gain/kernel_spike_visual_weight.pdf')
plt.close('all')