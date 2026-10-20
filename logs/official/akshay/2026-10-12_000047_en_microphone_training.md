# Demo run 2026-10-12 00:00:47 · English · microphone · training voice

**Result: 0 of 6 models dialled the correct number (0631123456).** Run status: complete. Words: 10 detected, 10 expected.

## 1. Run

| Item | Value |
|---|---|
| Run ID | 2026-10-12_000047_en_microphone_training |
| Started / ended | 2026-10-12T00:00:47+02:00 / 2026-10-12T00:01:12+02:00 |
| Language | English |
| Source | microphone |
| Voice (declared) | training voice |
| Speaker | not recorded (microphone) |
| Input device | Microphone Array (Intel® Smart  (44100 Hz) |
| Target number | 0631123456 |
| Audio length | 23.9 s |
| Training voices / test voice | akshay, maulik / tapan |
| Code | commit 3fdd983 on main |
| Python / scikit-learn / numpy | 3.12.10 / 1.6.1 / 2.5.4 |
| Platform | Windows-11-10.0.26300-SP0 |

## 2. Checklist of the brief

| # | Brief | Expected outcome | Result | What happened |
|---|---|---|---|---|
| C1 | §2 | The live loop runs from audio to dialled number. | **PASS** | Audio → event detection → features → classification → digit shown → number assembled. 10 word(s) went through the loop. |
| C2 | §2, §3 | Event detection finds every spoken digit (energy + zero-crossing rate). | **FAIL** | 10 word(s) detected for 10 digit(s). Run status: complete. Likely cause: digit 7: merged; word 6: extra sound or word. See section 3. |
| C3 | §2 | All three approaches classify each word live (default and tuned). | **PASS** | 6 models answered every word. |
| C4 | §2 | The digit is shown in real time. | **INFO** | The loop closes a word after 0.3 s of quiet. Then each model needs at most 11.6 ms to answer. |
| C5 | §1 | Pizza test: the dialled number equals the target (one wrong digit = wrong shop). | **FAIL** | Target 0631123456. Passed: none. Failed: 1.a default, 1.a tuned, 1.b default, 1.b tuned, 1.c default, 1.c tuned. |
| C6 | §5.2 | A new speaker is tested (no leakage of voices between training and test). | **INFO** | Declared as training voice. The system cannot check who speaks. Training voices: akshay, maulik; test voice: tapan. |
| C7 | §1, §7 | Per-digit errors compound: a number needs every digit right. | **INFO** | Chance of a correct 10-digit number from the test accuracy (one run cannot confirm it): 1.a default 0.3 %; 1.a tuned 1.2 %; 1.b default 16.1 %; 1.b tuned 14.6 %; 1.c default 4.5 %; 1.c tuned 5.6 %. Observed: 0 of 6 models correct. |
| C8 | §5.6 | 1.c reports accuracy AND data-size reduction against 1.b (same classifier). | **INFO** | tuned: 89.2× fewer values (144 instead of 12850), validation +3.0 pp, test -7.5 pp (both RBF SVM); default: not comparable, the classifiers differ (Linear SVM against RBF SVM) |
| C9 | §6 | The same code runs the simulation and the microphone. | **INFO** | Source: microphone. Both sources use the loop in live_demo/web.py (blocks of 0.1 s, threshold 4× the room noise, word ends after 3 quiet blocks). |
| C10 | Tuning Protocol | The loop settings and models stay the same during the pizza test. | **INFO** | Settings ID e91785ab (loop and event-extraction settings), models ID 73a1c2a1 (model files). Compare both with the other runs in summary.md. |

## 3. What worked and what failed

### Worked

- The loop heard 10 word(s) and every model answered each of them.
- 1.a default (RBF SVM): 7 of 9 detected words right.
- 1.a tuned (Linear SVM): 3 of 9 detected words right.
- 1.b default (Linear SVM): 9 of 9 detected words right.
- 1.b tuned (RBF SVM): 9 of 9 detected words right.
- 1.c default (RBF SVM): 8 of 9 detected words right.
- 1.c tuned (RBF SVM): 9 of 9 detected words right.

### Failed and why

- **Not heard: position 7: 3 (three).** Likely cause: merged. The gap (1.3 s) is not longer than the usual gap (1.1 s): there was no room for a separate word. All models are affected: they never received this word. More training does not help here.
- **Extra word 6** at 17.0 s. Likely cause: extra sound or word. A sound of 0.36 s (peak 22.3 dB) was taken as a word, but no target digit fits it: an extra word, or a noise that looks like speech.

**Wrong digits (classification), 9 in total.** 3 of them are confusions that the model also made on the test set; 6 did not occur on the test set. A wrong digit is a property of one model, not of the loop.

| Model | Position | Expected | Heard | Same confusion on the test set | Test accuracy of the expected digit | Word length (s) | Peak (dB) | Note |
|---|---|---|---|---|---|---|---|---|
| 1.a default | 1 | 0 (zero) | 4 | never (0 of 30) | 20 % | 0.58 | 65.3 | – |
| 1.a default | 9 | 5 (five) | 9 | never (0 of 10) | 90 % | 0.41 | 63.6 | – |
| 1.a tuned | 1 | 0 (zero) | 5 | never (0 of 30) | 63 % | 0.58 | 65.3 | – |
| 1.a tuned | 2 | 6 (six) | 0 | 3 of 10 words | 70 % | 0.76 | 58.2 | – |
| 1.a tuned | 3 | 3 (three) | 8 | 2 of 10 words | 40 % | 0.49 | 61.4 | – |
| 1.a tuned | 8 | 4 (four) | 1 | never (0 of 10) | 50 % | 0.51 | 61.8 | – |
| 1.a tuned | 9 | 5 (five) | 9 | never (0 of 10) | 80 % | 0.41 | 63.6 | – |
| 1.a tuned | 10 | 6 (six) | 0 | 3 of 10 words | 70 % | 0.84 | 58.9 | – |
| 1.c default | 9 | 5 (five) | 8 | never (0 of 10) | 30 % | 0.41 | 63.6 | – |


## 4. Word by word

| # | Expected | Heard as word no. | Start (s) | Length (s) | Peak (dB) | 1.a default | 1.a tuned | 1.b default | 1.b tuned | 1.c default | 1.c tuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 (zero) | 1 | 8.0 | 0.58 | 65.3 | 4 WRONG | 5 WRONG | 0 ok | 0 ok | 0 ok | 0 ok |
| 2 | 6 (six) | 2 | 10.0 | 0.76 | 58.2 | 6 ok | 0 WRONG | 6 ok | 6 ok | 6 ok | 6 ok |
| 3 | 3 (three) | 3 | 12.1 | 0.49 | 61.4 | 3 ok | 8 WRONG | 3 ok | 3 ok | 3 ok | 3 ok |
| 4 | 1 (one) | 4 | 14.0 | 0.42 | 65.1 | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok |
| 5 | 1 (one) | 5 | 15.7 | 0.42 | 67.4 | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok |
| 6 | 2 (two) | 7 | 17.8 | 0.43 | 60.1 | 2 ok | 2 ok | 2 ok | 2 ok | 2 ok | 2 ok |
| 7 | 3 (three) | – | – | – | – | not heard | not heard | not heard | not heard | not heard | not heard |
| 8 | 4 (four) | 8 | 19.6 | 0.51 | 61.8 | 4 ok | 1 WRONG | 4 ok | 4 ok | 4 ok | 4 ok |
| 9 | 5 (five) | 9 | 21.2 | 0.41 | 63.6 | 9 WRONG | 9 WRONG | 5 ok | 5 ok | 8 WRONG | 5 ok |
| 10 | 6 (six) | 10 | 22.8 | 0.84 | 58.9 | 6 ok | 0 WRONG | 6 ok | 6 ok | 6 ok | 6 ok |
| – | – | 6 | 17.0 | 0.36 | 22.3 | 3 | 0 | 6 | 6 | 8 | 3 |
| | **Dialled** | | | | | `4631132496` | `5081102190` | `0631162456` | `0631162456` | `0631182486` | `0631132456` |

## 5. Pizza test per model

| Model | Classifier | Dialled | Target | Pizza test | Digits right | Right among detected | Chance of a correct number (test accuracy ^ digits) |
|---|---|---|---|---|---|---|---|
| 1.a default | RBF SVM | `4631132496` | `0631123456` | **FAIL** | 7 of 10 | 7 of 9 | 0.3 % |
| 1.a tuned | Linear SVM | `5081102190` | `0631123456` | **FAIL** | 3 of 10 | 3 of 9 | 1.2 % |
| 1.b default | Linear SVM | `0631162456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 9 | 16.1 % |
| 1.b tuned | RBF SVM | `0631162456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 9 | 14.6 % |
| 1.c default | RBF SVM | `0631182486` | `0631123456` | **FAIL** | 8 of 10 | 8 of 9 | 4.5 % |
| 1.c tuned | RBF SVM | `0631132456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 9 | 5.6 % |

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
| calibration_noise_rms | 1.7e-05 |
| threshold_rms | 6.8e-05 |
| room_during_run_db | -1.2 |

## 8. Level of every 0.1 s block

Each character is 4 dB above the room noise at the calibration (▁ = 0 to 4 dB, █ = 28 dB or more). The threshold is 12.0 dB (▄ and higher). `#` = word, `x` = loud sound without a word.

```
  0.0 s  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
                                                           
  5.0 s  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██████▇▄▂▁▁▁▁▁▁▁▁▁▁▁
                                       ########            
 10.0 s  ▇█████▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████▃▁▁▁▁▁▁▁▁▁▁▁▁▁█████▄▁▁▁▁
         ######               #####              ######    
 15.0 s  ▁▁▁▁▁▁▁█████▄▁▁▁▁▁▁▁▆▄▃▁▁▁▁▁████▇▁▁▁▁▁▁▁▁▁▁▁▁▁▇███
                ######       ##      #####             ####
 20.0 s  █▆▁▁▁▁▁▁▁▁▁▂█████▅▁▁▁▁▁▁▁▁▁▁▅█████▇▄▁▁▁
         ##          ######          ########   
```
