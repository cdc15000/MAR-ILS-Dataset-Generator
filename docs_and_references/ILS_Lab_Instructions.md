# Laboratory Instructions

| | |
|---|---|
| Study | ASTM WK101378, draft Revision 06: interlaboratory study |
| Version | 1.0 draft, 2026-11-12 |
| Applies to | ILS protocol §A8.2, Steps 1 to 6 |

These instructions give the commands for each step of the protocol. The protocol governs; where the two differ, follow the protocol and tell the ILS Coordinator.

You run the supplied software as it is. Do not modify the software, its settings or the datasets (protocol §A8.1).

---

## 1. What You Need

| Item | Requirement |
|---|---|
| Computer | Any 64-bit computer running Linux, macOS or Windows |
| Memory | 16 GB recommended |
| Disk | About 150 GB free, for the datasets and the output |
| Python | 3.11 or later |
| Packages | Those listed in `requirements.txt` |

No CT scanner, GPU or commercial software is needed.

## 2. Install the Software

1. Download the release of the software named in the coordinator's letter, and unpack it. All commands below are run from the unpacked directory.

2. Create a Python environment and install the packages:

   ```bash
   python3 -m venv ils-env
   source ils-env/bin/activate          # Windows: ils-env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Print the versions, and keep the output. The questionnaire asks for it (Attachment 3, item 5):

   ```bash
   python run_control_v9_0.py --versions
   ```

## 3. Step 1: Download and Verify the Datasets

1. Download your three replicate datasets, the null-check dataset `NULL` and the familiarization dataset `FAMILIARIZATION`, using the access link sent by ILS staff. Your replicate datasets are named in the coordinator's letter. In the commands below they are written `REP_A`, `REP_B` and `REP_C`; use the names you were given.

2. Verify the checksums of each dataset:

   ```bash
   python run_control_v9_0.py --verify FAMILIARIZATION
   python run_control_v9_0.py --verify REP_A
   python run_control_v9_0.py --verify REP_B
   python run_control_v9_0.py --verify REP_C
   python run_control_v9_0.py --verify NULL
   ```

   Each command ends with `PASS` or `FAIL`. If a dataset fails, download it again. If it fails again, contact the coordinator.

3. Each replicate dataset has 8,000 sinograms in `sinograms/` and 8,000 images in `noMAR/`. The null-check dataset has 4,000 of each. The verification of item 2 confirms that every file is present.

## 4. Step 2: Installation Check

Do this step before you process any study dataset.

1. Reconstruction check (draft standard §12.2):

   ```bash
   python run_control_v9_0.py --recon-check FAMILIARIZATION/recon_check/recon_check.h5
   ```

   The command prints the largest difference from the supplied image, in HU, and `PASS` or `FAIL`. Write the largest difference on the return form.

2. Process the familiarization dataset and compare the result with the expected images supplied with it:

   ```bash
   python run_control_v9_0.py FAMILIARIZATION C1_FAM
   python run_control_v9_0.py --compare C1_FAM FAMILIARIZATION/expected_C1
   ```

   The second command ends with `PASS` if every image equals its expected image, pixel for pixel. Write the result on the return form.

3. If either check fails, do not go on. Send the printed output and the output of `--versions` to the coordinator.

## 5. Step 3: Processing

Run the supplied improving control on each of your datasets. Each command reads every sinogram of the dataset and writes one DICOM image per realization to the run directory.

```bash
python run_control_v9_0.py REP_A C1_R1
python run_control_v9_0.py REP_B C1_R2
python run_control_v9_0.py REP_C C1_R3
python run_control_v9_0.py NULL  C1_NULL
```

- Use the run directory names shown. `C1_R1`, `C1_R2` and `C1_R3` follow the order of the datasets in the coordinator's letter.
- A run prints its progress and the time remaining. If a run is interrupted, give the same command again; images already written are kept.
- The software uses all but one of the computer's processors. To use fewer, add `--workers N`.
- Each run writes a record beside the run directory, for example `C1_R1_run_record.json`. Keep it with the run directory.

## 6. Step 4: Scoring

1. Score each run directory against the dataset it was produced from:

   ```bash
   python run_cho_analysis_v9_0.py --dataset-dir REP_A --mar-output-dir C1_R1
   python run_cho_analysis_v9_0.py --dataset-dir REP_B --mar-output-dir C1_R2
   python run_cho_analysis_v9_0.py --dataset-dir REP_C --mar-output-dir C1_R3
   python run_cho_analysis_v9_0.py --dataset-dir NULL  --mar-output-dir C1_NULL
   ```

   Each command writes `<run directory>_cho_results.json` beside the run directory and prints, for each signal level, AUC∞ of the noMAR series, AUC∞ of the MAR series and ΔAUC∞, each with its error.

2. Check the printed output:

   - If a line beginning `WARNING` says that the noMAR AUC∞ differs from the reference by more than 0.005, stop and contact the coordinator.
   - For `C1_NULL`, the line `null-signal check` should say `within 0.02 of 0.533`. If it says `OUTSIDE`, stop and contact the coordinator (draft standard §12.3).

## 7. Step 5: Submission

Within the execution window, send to the ILS Coordinator:

- `C1_R1_cho_results.json`, `C1_R2_cho_results.json`, `C1_R3_cho_results.json` and `C1_NULL_cho_results.json`
- the questionnaire (Attachment 3)
- your log of unusual events

Report the results on the ASTM data report form and send it, with the return form (Attachment 2), to ILS staff at ILS@astm.org.

Do not send images or sinograms.

## 8. Step 6: Retention

Keep the run directories and their run records for 18 months from the date of submission.

## 9. Pilot Laboratories Only

Pilot laboratories also apply the supplied degrading control to the control dataset (protocol §B4):

```bash
python run_control_v9_0.py --verify CONTROL
python run_control_v9_0.py CONTROL C1_CONTROL --control degrading
python run_cho_analysis_v9_0.py --dataset-dir CONTROL --mar-output-dir C1_CONTROL
```

Send `C1_CONTROL_cho_results.json` with the other results.

## 10. Help

Direct all questions about the study to the ILS Coordinator, at the address in the coordinator's letter. When you report a problem, include the command, its printed output and the output of `python run_control_v9_0.py --versions`.
