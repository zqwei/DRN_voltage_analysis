# DRN_voltage_analysis

## variables after postprocessing

### `Voltr_raw.npz`
* `A_`: spatial component matrix -- nPixel x nComponents
* `C_`: temporal component matrix -- nComponents x nTimepoints
* `base_`: "baseline" component matrix -- nComponents x 1
* `df/f`: $C_/(<C_> + base_ - background)$

### `Voltr_spikes.npz`
* `voltrs`: `df/f`
* `voltr_`: nomarlized `df/f` (z-score using a running mean and std.)
* `spk1`: type 1 spike in neural network detection at frame t (big amplitude) 
* `spk2`: type 2 spike in neural network detection if any (small amplitude) 
* `spk`: sume of spk1 and spk2  -- nComponents x nTimepoints; 0 no spike; 1 spike
* `spkprob`: probability of spikes at frame t  -- nComponents x nTimepoints

### Voltr_subvolt.npz
* `subvolt_`: subthreshold voltage computing using `df/f`   -- nComponents x nTimepoints 
* `norm_subvolt`: subthreshold voltage computing using nomalized `df/f`  -- nComponents x nTimepoints
