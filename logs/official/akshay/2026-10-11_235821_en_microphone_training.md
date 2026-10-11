# Demo run 2026-10-11 23:58:21 · English · microphone · training voice

**Result: 0 of 6 models dialled the correct number (0631123456).** Run status: complete. Words: 10 detected, 10 expected.

## 1. Run

| Item | Value |
|---|---|
| Run ID | 2026-10-11_235821_en_microphone_training |
| Started / ended | 2026-10-11T23:58:21+02:00 / 2026-10-11T23:58:37+02:00 |
| Language | English |
| Source | microphone |
| Voice (declared) | training voice |
| Speaker | not recorded (microphone) |
| Input device | Microphone Array (Intel® Smart  (44100 Hz) |
| Target number | 0631123456 |
| Audio length | 15.5 s |
| Training voices / test voice | akshay, maulik / tapan |
| Code | commit 3fdd983 on main |
| Python / scikit-learn / numpy | 3.12.10 / 1.6.1 / 2.5.4 |
| Platform | Windows-11-10.0.26300-SP0 |

## 2. Checklist of the brief

| # | Brief | Expected outcome | Result | What happened |
|---|---|---|---|---|
| C1 | §2 | The live loop runs from audio to dialled number. | **PASS** | Audio → event detection → features → classification → digit shown → number assembled. 10 word(s) went through the loop. |
| C2 | §2, §3 | Event detection finds every spoken digit (energy + zero-crossing rate). | **PASS** | 10 word(s) detected for 10 digit(s). |
| C3 | §2 | All three approaches classify each word live (default and tuned). | **PASS** | 6 models answered every word. |
| C4 | §2 | The digit is shown in real time. | **INFO** | The loop closes a word after 0.3 s of quiet. Then each model needs at most 13.3 ms to answer. |
| C5 | §1 | Pizza test: the dialled number equals the target (one wrong digit = wrong shop). | **FAIL** | Target 0631123456. Passed: none. Failed: 1.a default, 1.a tuned, 1.b default, 1.b tuned, 1.c default, 1.c tuned. |
| C6 | §5.2 | A new speaker is tested (no leakage of voices between training and test). | **INFO** | Declared as training voice. The system cannot check who speaks. Training voices: akshay, maulik; test voice: tapan. |
| C7 | §1, §7 | Per-digit errors compound: a number needs every digit right. | **INFO** | Chance of a correct 10-digit number from the test accuracy (one run cannot confirm it): 1.a default 0.3 %; 1.a tuned 1.2 %; 1.b default 16.1 %; 1.b tuned 14.6 %; 1.c default 4.5 %; 1.c tuned 5.6 %. Observed: 0 of 6 models correct. |
| C8 | §5.6 | 1.c reports accuracy AND data-size reduction against 1.b (same classifier). | **INFO** | tuned: 89.2× fewer values (144 instead of 12850), validation +3.0 pp, test -7.5 pp (both RBF SVM); default: not comparable, the classifiers differ (Linear SVM against RBF SVM) |
| C9 | §6 | The same code runs the simulation and the microphone. | **INFO** | Source: microphone. Both sources use the loop in live_demo/web.py (blocks of 0.1 s, threshold 4× the room noise, word ends after 3 quiet blocks). |
| C10 | Tuning Protocol | The loop settings and models stay the same during the pizza test. | **INFO** | Settings ID e91785ab (loop and event-extraction settings), models ID 73a1c2a1 (model files). Compare both with the other runs in summary.md. The room was 6.0 dB quieter during the run than at the calibration, so the threshold was 6.0 dB higher than planned. |

## 3. What worked and what failed

### Worked

- The loop heard 10 word(s) and every model answered each of them.
- 1.a default (RBF SVM): 8 of 10 detected words right.
- 1.a tuned (Linear SVM): 4 of 10 detected words right.
- 1.b default (Linear SVM): 9 of 10 detected words right.
- 1.b tuned (RBF SVM): 9 of 10 detected words right.
- 1.c default (RBF SVM): 9 of 10 detected words right.
- 1.c tuned (RBF SVM): 9 of 10 detected words right.

### Failed and why


**Wrong digits (classification), 12 in total.** 5 of them are confusions that the model also made on the test set; 7 did not occur on the test set. A wrong digit is a property of one model, not of the loop.

| Model | Position | Expected | Heard | Same confusion on the test set | Test accuracy of the expected digit | Word length (s) | Peak (dB) | Note |
|---|---|---|---|---|---|---|---|---|
| 1.a default | 1 | 0 (zero) | 1 | never (0 of 30) | 20 % | 0.35 | 47.5 | – |
| 1.a default | 10 | 6 (six) | 3 | never (0 of 10) | 100 % | 0.94 | 55.3 | – |
| 1.a tuned | 1 | 0 (zero) | 1 | 1 of 30 words | 63 % | 0.35 | 47.5 | – |
| 1.a tuned | 3 | 3 (three) | 8 | 2 of 10 words | 40 % | 0.55 | 54.8 | – |
| 1.a tuned | 6 | 2 (two) | 1 | never (0 of 10) | 50 % | 0.51 | 56.1 | – |
| 1.a tuned | 8 | 4 (four) | 1 | never (0 of 10) | 50 % | 0.46 | 56.8 | – |
| 1.a tuned | 9 | 5 (five) | 9 | never (0 of 10) | 80 % | 0.53 | 56.9 | – |
| 1.a tuned | 10 | 6 (six) | 0 | 3 of 10 words | 70 % | 0.94 | 55.3 | – |
| 1.b default | 1 | 0 (zero) | 8 | never (0 of 30) | 83 % | 0.35 | 47.5 | – |
| 1.b tuned | 1 | 0 (zero) | 8 | never (0 of 30) | 83 % | 0.35 | 47.5 | – |
| 1.c default | 1 | 0 (zero) | 2 | 1 of 30 words | 77 % | 0.35 | 47.5 | – |
| 1.c tuned | 1 | 0 (zero) | 2 | 1 of 30 words | 77 % | 0.35 | 47.5 | – |

- **Calibration.** The room level during the run was -6.0 dB against the calibration noise. Calibrate again in the room where you speak.

## 4. Word by word

| # | Expected | Heard as word no. | Start (s) | Length (s) | Peak (dB) | 1.a default | 1.a tuned | 1.b default | 1.b tuned | 1.c default | 1.c tuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 (zero) | 1 | 0.1 | 0.35 | 47.5 | 1 WRONG | 1 WRONG | 8 WRONG | 8 WRONG | 2 WRONG | 2 WRONG |
| 2 | 6 (six) | 2 | 2.1 | 0.86 | 54.6 | 6 ok | 6 ok | 6 ok | 6 ok | 6 ok | 6 ok |
| 3 | 3 (three) | 3 | 3.9 | 0.55 | 54.8 | 3 ok | 8 WRONG | 3 ok | 3 ok | 3 ok | 3 ok |
| 4 | 1 (one) | 4 | 5.4 | 0.49 | 59.5 | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok |
| 5 | 1 (one) | 5 | 6.8 | 0.54 | 61.9 | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok |
| 6 | 2 (two) | 6 | 8.4 | 0.51 | 56.1 | 2 ok | 1 WRONG | 2 ok | 2 ok | 2 ok | 2 ok |
| 7 | 3 (three) | 7 | 9.9 | 0.56 | 54.7 | 3 ok | 3 ok | 3 ok | 3 ok | 3 ok | 3 ok |
| 8 | 4 (four) | 8 | 11.4 | 0.46 | 56.8 | 4 ok | 1 WRONG | 4 ok | 4 ok | 4 ok | 4 ok |
| 9 | 5 (five) | 9 | 13.0 | 0.53 | 56.9 | 5 ok | 9 WRONG | 5 ok | 5 ok | 5 ok | 5 ok |
| 10 | 6 (six) | 10 | 14.5 | 0.94 | 55.3 | 3 WRONG | 0 WRONG | 6 ok | 6 ok | 6 ok | 6 ok |
| | **Dialled** | | | | | `1631123453` | `1681113190` | `8631123456` | `8631123456` | `2631123456` | `2631123456` |

## 5. Pizza test per model

| Model | Classifier | Dialled | Target | Pizza test | Digits right | Right among detected | Chance of a correct number (test accuracy ^ digits) |
|---|---|---|---|---|---|---|---|
| 1.a default | RBF SVM | `1631123453` | `0631123456` | **FAIL** | 8 of 10 | 8 of 10 | 0.3 % |
| 1.a tuned | Linear SVM | `1681113190` | `0631123456` | **FAIL** | 4 of 10 | 4 of 10 | 1.2 % |
| 1.b default | Linear SVM | `8631123456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 10 | 16.1 % |
| 1.b tuned | RBF SVM | `8631123456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 10 | 14.6 % |
| 1.c default | RBF SVM | `2631123456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 10 | 4.5 % |
| 1.c tuned | RBF SVM | `2631123456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 10 | 5.6 % |

## 6. Models (same values as the page)

| Model | Classifier | k | Values per word | Validation | Test | Test words | Model file (SHA-256, first 12) | Results file |
|---|---|---|---|---|---|---|---|---|
| 1.a default | RBF SVM | – | 30 | 55.0 % | 55.8 % | 120 | models/1a_en_best.joblib (656f26bc308d) | results/1a_en_rbf_svm.json |
| 1.a tuned | Linear SVM | – | 10 | 60.8 % | 64.2 % | 120 | models/1a_en_tuned_best.joblib (9a210895ff90) | results/1a_en_tuned_linear_svm.json |
| 1.b default | Linear SVM | – | 12850 | 70.1 % | 83.3 % | 120 | models/1b_en_best.joblib (d83042eaa9a7) | results/1b_en_linear_svm.json |
| 1.b tuned | RBF SVM | – | 12850 | 71.1 % | 82.5 % | 120 | models/1b_en_tuned_best.joblib (813e19b2230d) | results/1b_en_tuned_rbf_svm.json |
| 1.c default | RBF SVM | 16 | 256 | 68.6 % | 73.3 % | 120 | models/1c_en_best.joblib (8b03e066ef31) | results/1c_en_rbf_svm.json |
| 1.c tuned | RBF SVM | 12 | 144 | 74.2 % | 75.0 % | 120 | models/1c_en_tuned_best.joblib (505b3a71e596) | results/1c_en_tuned_rbf_svm.json |

## 7. Settings of the loop

| Item | Value |
|---|---|
| settings_id | e91785ab |
| models_id | 73a1c2a1 |
| block_s | 0.1 |
| quiet_blocks | 3 |
| loudness_factor | 4.0 |
| sample_rate | 16000 |
| frame_ms | 25 |
| hop_ms | 10 |
| pad_ms | 50 |
| noise_percentile | 10 |
| energy_factor | 4.0 |
| peak_fraction | 0.05 |
| max_gap_ms | 250 |
| max_word_ms | 1200 |
| zcr_threshold | 0.3 |
| zcr_energy_factor | 2.0 |
| hiss_extend_ms | 200 |
| threshold_db | 12.0 |
| calibration_noise_rms | 3.2e-05 |
| threshold_rms | 0.000127 |
| room_during_run_db | -6.0 |

## 8. Level of every 0.1 s block

Each character is 4 dB above the room noise at the calibration (▁ = 0 to 4 dB, █ = 28 dB or more). The threshold is 12.0 dB (▄ and higher). `#` = word, `x` = loud sound without a word.

```
  0.0 s  ▁███▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▇█████▇▁▁▁▁▁▁▁▁▁▁▁█████▃▄▁▁▁▁
          ####                #######           #####      
  5.0 s  ▁▁▁▁█████▄▁▁▁▁▁▁▁▁█████▅▂▄▁▁▁▁▁▁▁▁████▆▁▁▁▁▁▁▁▁▂▁▇
             ######        ########        #####          #
 10.0 s  ████▅▁▂▁▁▁▁▁▁▁▅█████▄▁▁▁▁▁▁▁▁▃▆████▇▃▁▁▁▁▁▁▁▂█████
         #####         #######         ######         #####
 15.0 s  ██▂▁▂
         ##   
```
