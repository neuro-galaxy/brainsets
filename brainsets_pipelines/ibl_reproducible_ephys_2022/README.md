# ibl_reproducible_ephys_2022

Repeatedly inserted Neuropixels multi-electrode probes targeting the same brain locations (called the repeated site, including posterior parietal cortex, hippocampus, and thalamus) in mice performing a behavioral task. Mice were trained to use a steering wheel to indicate the position of visual stimuli. In the **basic task**, the probability of a stimulus appearing on the left or the right was equal(50:50). In the **full task**, the probability of stimuli appearing on the left vs. right switched in blocks of trials between 20:80 and 80:20.

Publication:
- International Brain Laboratory, Banga Kush, Benson Julius, Bhagat Jai, Biderman Dan, Birman Daniel, Bonacchi Niccolò, Bruijns Sebastian A, Buchanan Kelly, Campbell Robert AA, Carandini Matteo, Chapuis Gaëlle A, Churchland Anne K, Davatolhagh M Felicia, Lee Hyun Dong, Faulkner Mayo, Gerçek Berk, Hu Fei, Huntenburg Julia, Hurwitz Cole, Khanal Anup, Krasniak Christopher, Langfield Christopher, Meijer Guido T, Miska Nathaniel J, Mohammadi Zeinab, Noel Jean-Paul, Paninski Liam, Pan-Vazquez Alejandro, Roth Noam, Schartner Michael, Socha Karolina, Steinmetz Nicholas A, Svoboda Karel, Taheri Marsa, Urai Anne E, Wells Miles, West Steven J, Whiteway Matthew R, Winter Olivier, Witten Ilana B (2024) Reproducibility of in vivo electrophysiological measurements in mice eLife 13:RP100840. https://doi.org/10.7554/eLife.100840.1.

Subject(s)
- 83 mice aged 111 - 442 days.

Neural variables
- Spike-sorted neural activity

Task Variables
- Choice: The direction in which the mouse turned the steering wheel (left or right; discrete).
- Block prior: The block identity which determines the probability of stimuli appearing on the left vs. right (discrete).
- Wheel speed: The velocity at which the mouse moved the steering wheel (continuous).
- Whisker motion energy: A measure of whisker movement activity (continuous).

This pipeline requires `ONE-api` and `ibllib` to be installed.

```bash
pip install ONE-api
pip install ibllib
```

To process a single session, run the following command:

```bash
python3 prepare_data.py --eid db4df448-e449-4a6f-a0e7-288711e7a75a
```
