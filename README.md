# Not all theta is real

**Removing illusory induced oscillations with Woody filtering and single-trial regression**
Zayneb Bellatif¹˒², James Bonaiuto¹˒², Máté Gyurkovics³, Quentin Moreau¹˒²

¹ Marc Jeannerod Institute of Cognitive Science (ISC), CNRS UMR5229, Bron, France ·
² Université Claude Bernard Lyon 1, Lyon, France ·
³ School of Psychology, University of East Anglia, Norwich, UK

---

## Repository layout

```
ern_theta/                 Shared functions
  woody.py                 Woody and stochastic Woody filters
  preprocessing.py         align_trials, regress_erp, reinstate_lags
  tfr.py                   Morlet power, baseline correction, band power
  simulation.py            Simulated ERN + Pe trials in pink noise
  plotting.py              Figures shared by the simulation notebooks
  config.py, io.py         Data paths, event codes, subject lists, epoch loading
simulations/
  00_erp_in_tf_intro.ipynb             Introduction: ERPs look like oscillations in TF space
  01_ern_simulation.ipynb              ERN simulations: jitter, regression, Woody
  01b_ern_simulation_noiseless.ipynb   Noiseless sanity check of Woody lag recovery
empirical_analyses/
  02_error_vs_correct.ipynb            Validation on empirical data (response-locked ERN)
  03_congruent_vs_incongruent.ipynb    Stimulus-locked N2: congruent vs incongruent
```
___

### Empirical data

The empirical notebooks use the preprocessed Flanker epochs from the **#EEGManyAnalysts** project (<https://www.coscience-personality.com/manyanalysts>), which are not redistributed here.


The Python Woody filter is a translation of the MATLAB implementation used in Bonaiuto et al. (2021), *Laminar dynamics of high amplitude beta bursts in human motor cortex*.
