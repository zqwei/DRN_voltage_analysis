# DRN Imaging Processing

## Processing pipeline
* `python DRN_processing_jobs.py pixel_denoise`
* `python DRN_processing_jobs.py registration`
* `python DRN_processing_jobs.py video_detrend`
* `python DRN_processing_jobs.py local_pca`
* `python DRN_processing_jobs.py demix_components` (default -- demix only using middle 1/3 of the time series)

## Variables after postprocessing

### Voltr_raw.npz
* `A_`: spatial component matrix -- nPixel x nComponents
* `C_`: temporal component matrix -- nComponents x nTimepoints
* `base_`: "baseline" component matrix -- nComponents x 1
* `df/f`: $C/(<C> + base - background)$

### Voltr_spikes.npz
* `voltrs`: `df/f`
* `voltr_`: nomarlized `df/f` (z-score using a running mean and std.)
* `spk1`: type 1 spike in neural network detection at frame t (big amplitude)
* `spk2`: type 2 spike in neural network detection if any (small amplitude)
* `spk`: sume of spk1 and spk2  -- nComponents x nTimepoints; 0 no spike; 1 spike
* `spkprob`: probability of spikes at frame t  -- nComponents x nTimepoints

### Voltr_subvolt.npz
* `subvolt_`: subthreshold voltage computing using `df/f`   -- nComponents x nTimepoints
* `norm_subvolt`: subthreshold voltage computing using nomalized `df/f`  -- nComponents x nTimepoints
