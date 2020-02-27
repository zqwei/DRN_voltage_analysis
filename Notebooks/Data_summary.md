## Behaviors
* (**Can TK fill this part?**)

### GA+Replay
### GA+Replay with DRN ablation (in plan?)

## Spike and subvolt

### Gain Adaptation (GA)
* Single cell example -- swim, raster plot, spike, subvolt
* Population dynamics by peak time -- 494 cells, 21 fish
* Selective neuron average -- 108 cells, 17 fish
* Motor clamp (selevtive neurons) -- 19 cells, 4 fish

### Kernel fits
* Spike: 494 cells
* Subvolt: 494 cells

### GA+Replay
* None
* **Do we need this?**

### SOVO
* Behavior example from `['05022019', 'Fish2-1-swimonly_visualonly']`
* Selective neuron average: 21 cells, 6 fish

### Population
* GA: 20 sessions, 15 fish
* Memory: 11 sessions, 8 fish
* GA+Ablt: 25 sessions, 15 fish
* GA+Ablt+control: 20 sessions, 10 fish

### Gaba Ablation
* Selective neuron average -- dynamics of spikes before/after ablation
* Selective neuron average -- normalized subvolt (dff) before/after ablation
* control: 57 cells, 10 fish
* ablt: 48 cells, 9 fish
* analysis for spike shape -- if any change before and after ablation


### Gaba Ablation Behaviors
* degree of adapatation, swim power, swim length
* control: 8 fish
* ablt: 12 fish


## SnFR example sessions
### First peak of GABA and Glu release are correlated with swim vigor
* GABA in GA task
    * Example recording session
    * dFF vs vel: SpearmanrResult(correlation=0.2652916073968705, pvalue=0.11253461250288342)
    * dFF vs swim: SpearmanrResult(correlation=0.46181753356771355, pvalue=1.696806854832783e-06)
* Glu in GA task
    * Example recording session
    * dFF vs vel: SpearmanrResult(correlation=0.14786023732119238, pvalue=0.037631589644074995)
    * dFF vs swim: SpearmanrResult(correlation=0.6869998353558286, pvalue=5.451578330971371e-29)
    
### Second peak of Glu release are correlated with visual feedback
* Glu in Random Gain (RG) task (motor-clamp trials)
    * Example recording session
    * dFF 2nd peak vs vel: SpearmanrResult(correlation=0.5143954480796585, pvalue=2.347997075262531e-06)
* Glu in Random Delay (RD) task (motor-clamp trials)
    * Example recording session (the same as the RG task)
    * dFF 2nd peak time vs vel onset time: F_onewayResult(statistic=54.78951858020245, pvalue=3.396881689204097e-17)

## GABASnFR
### GA

### GA + RG -- encoding of swim vigor vs visual feedback
* 
* 

## GluSnFR
### GA
### RG-only
### RD
### SOVO
### GA+RG
### GA+RD -- visual feedback
### GA+SOVO
### GA+Replay
* Glu: 6 planes, 6 fish


## Model

## RNAseq + HCN-KO

## Single-cell tracing + upstream
* (**Can TK fill this part?**)


# Manuscript outline
## Figures
### Figure 1
* Replay per se does not create memory (half-solid data)
* This swim-dependence is serotonin dependent (no data)

### Figure 2
* Voltage imaging of serotonin neurons
* DRN encodes visual feedback gain
* DRN neurons shows gated response  during the above task (no data, we’ll use SOVO as a replacement for a while) at single-cell and population levels.

### Figure 3
* Glutamate input + their visual encoding
* GABA input + their motor encoding
* visual input is coming from preoptic area (or whatever it is)

### Figure 4
* Visually-driven glutamate input is not gated
* Upstream region is not gated

### Figure 5
* Ablation of local GABA neurons impairs visual encoding
* Model confirms the gating effect.
* (for Fig 5) Analysis of HCN4l-knockout fish.
* (for Fig 5) Optogenetic activation. This may be for revision.















