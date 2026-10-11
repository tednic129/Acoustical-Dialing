# Demo run 2026-10-11 23:34:32 · German · recordings (simulation) · training voice

**Result: 5 of 6 models dialled the correct number (0631123456).** Run status: complete. Words: 10 detected, 10 expected.

## 1. Run

| Item | Value |
|---|---|
| Run ID | 2026-10-11_233432_de_simulation_training |
| Started / ended | 2026-10-11T23:34:32+02:00 / 2026-10-11T23:34:46+02:00 |
| Language | German |
| Source | recordings (simulation) |
| Voice (declared) | training voice |
| Speaker | akshay |
| Input device | – |
| Target number | 0631123456 |
| Audio length | 13.3 s |
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
| C4 | §2 | The digit is shown in real time. | **INFO** | The loop closes a word after 0.3 s of quiet. Then each model needs at most 18.5 ms to answer. |
| C5 | §1 | Pizza test: the dialled number equals the target (one wrong digit = wrong shop). | **FAIL** | Target 0631123456. Passed: 1.a default, 1.b default, 1.b tuned, 1.c default, 1.c tuned. Failed: 1.a tuned. |
| C6 | §5.2 | A new speaker is tested (no leakage of voices between training and test). | **INFO** | Training voice (akshay): the models saw this voice. This run checks the loop, not speaker independence. |
| C7 | §1, §7 | Per-digit errors compound: a number needs every digit right. | **INFO** | Chance of a correct 10-digit number from the test accuracy (one run cannot confirm it): 1.a default 0.3 %; 1.a tuned 0.0 %; 1.b default 8.7 %; 1.b tuned 6.3 %; 1.c default 1.2 %; 1.c tuned 5.6 %. Observed: 5 of 6 models correct. |
| C8 | §5.6 | 1.c reports accuracy AND data-size reduction against 1.b (same classifier). | **INFO** | tuned: 200.8× fewer values (64 instead of 12850), validation -0.6 pp, test -0.8 pp (both RBF SVM); default: not comparable, the classifiers differ (Linear SVM against RBF SVM) |
| C9 | §6 | The same code runs the simulation and the microphone. | **INFO** | Source: recordings (simulation). Both sources use the loop in live_demo/web.py (blocks of 0.1 s, threshold 4× the room noise, word ends after 3 quiet blocks). |
| C10 | Tuning Protocol | The loop settings and models stay the same during the pizza test. | **INFO** | Settings ID e91785ab (loop and event-extraction settings), models ID c0a27a09 (model files). Compare both with the other runs in summary.md. |

## 3. What worked and what failed

### Worked

- The loop heard 10 word(s) and every model answered each of them.
- 1.a default (Linear SVM): 10 of 10 detected words right.
- 1.a tuned (Linear SVM): 9 of 10 detected words right.
- 1.b default (Linear SVM): 10 of 10 detected words right.
- 1.b tuned (RBF SVM): 10 of 10 detected words right.
- 1.c default (RBF SVM): 10 of 10 detected words right.
- 1.c tuned (RBF SVM): 10 of 10 detected words right.

### Failed and why


**Wrong digits (classification), 1 in total.** 1 of them are confusions that the model also made on the test set; 0 did not occur on the test set. A wrong digit is a property of one model, not of the loop.

| Model | Position | Expected | Heard | Same confusion on the test set | Test accuracy of the expected digit | Word length (s) | Peak (dB) | Note |
|---|---|---|---|---|---|---|---|---|
| 1.a tuned | 4 | 1 (eins) | 2 | 6 of 10 words | 20 % | 0.72 | 39.5 | – |


## 4. Word by word

| # | Expected | Heard as word no. | Start (s) | Length (s) | Peak (dB) | 1.a default | 1.a tuned | 1.b default | 1.b tuned | 1.c default | 1.c tuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 (null) | 1 | 1.0 | 0.68 | 42.4 | 0 ok | 0 ok | 0 ok | 0 ok | 0 ok | 0 ok |
| 2 | 6 (sechs) | 2 | 2.1 | 0.68 | 42.1 | 6 ok | 6 ok | 6 ok | 6 ok | 6 ok | 6 ok |
| 3 | 3 (drei) | 3 | 3.4 | 0.71 | 44.7 | 3 ok | 3 ok | 3 ok | 3 ok | 3 ok | 3 ok |
| 4 | 1 (eins) | 4 | 4.7 | 0.72 | 39.5 | 1 ok | 2 WRONG | 1 ok | 1 ok | 1 ok | 1 ok |
| 5 | 1 (eins) | 5 | 5.9 | 0.68 | 40.2 | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok | 1 ok |
| 6 | 2 (zwei) | 6 | 7.3 | 0.83 | 40.1 | 2 ok | 2 ok | 2 ok | 2 ok | 2 ok | 2 ok |
| 7 | 3 (drei) | 7 | 8.7 | 0.77 | 41.6 | 3 ok | 3 ok | 3 ok | 3 ok | 3 ok | 3 ok |
| 8 | 4 (vier) | 8 | 10.0 | 0.56 | 43.5 | 4 ok | 4 ok | 4 ok | 4 ok | 4 ok | 4 ok |
| 9 | 5 (fünf) | 9 | 11.2 | 0.49 | 43.7 | 5 ok | 5 ok | 5 ok | 5 ok | 5 ok | 5 ok |
| 10 | 6 (sechs) | 10 | 12.3 | 0.77 | 40.9 | 6 ok | 6 ok | 6 ok | 6 ok | 6 ok | 6 ok |
| | **Dialled** | | | | | `0631123456` | `0632123456` | `0631123456` | `0631123456` | `0631123456` | `0631123456` |

## 5. Pizza test per model

| Model | Classifier | Dialled | Target | Pizza test | Digits right | Right among detected | Chance of a correct number (test accuracy ^ digits) |
|---|---|---|---|---|---|---|---|
| 1.a default | Linear SVM | `0631123456` | `0631123456` | **PASS** | 10 of 10 | 10 of 10 | 0.3 % |
| 1.a tuned | Linear SVM | `0632123456` | `0631123456` | **FAIL** | 9 of 10 | 9 of 10 | 0.0 % |
| 1.b default | Linear SVM | `0631123456` | `0631123456` | **PASS** | 10 of 10 | 10 of 10 | 8.7 % |
| 1.b tuned | RBF SVM | `0631123456` | `0631123456` | **PASS** | 10 of 10 | 10 of 10 | 6.3 % |
| 1.c default | RBF SVM | `0631123456` | `0631123456` | **PASS** | 10 of 10 | 10 of 10 | 1.2 % |
| 1.c tuned | RBF SVM | `0631123456` | `0631123456` | **PASS** | 10 of 10 | 10 of 10 | 5.6 % |

## 6. Models (same values as the page)

| Model | Classifier | k | Values per word | Validation | Test | Test words | Model file (SHA-256, first 12) | Results file |
|---|---|---|---|---|---|---|---|---|
| 1.a default | Linear SVM | – | 30 | 52.2 % | 55.8 % | 120 | models/1a_best.joblib (b676c060cf0b) | results/1a_linear_svm.json |
| 1.a tuned | Linear SVM | – | 10 | 53.9 % | 43.3 % | 120 | models/1a_tuned_best.joblib (aebec305f126) | results/1a_tuned_linear_svm.json |
| 1.b default | Linear SVM | – | 12850 | 65.1 % | 78.3 % | 120 | models/1b_best.joblib (84d7c53e51da) | results/1b_linear_svm.json |
| 1.b tuned | RBF SVM | – | 12850 | 66.1 % | 75.8 % | 120 | models/1b_tuned_best.joblib (188a897e540e) | results/1b_tuned_rbf_svm.json |
| 1.c default | RBF SVM | 16 | 256 | 60.4 % | 64.2 % | 120 | models/1c_best.joblib (972dfbc327e4) | results/1c_rbf_svm.json |
| 1.c tuned | RBF SVM | 8 | 64 | 65.5 % | 75.0 % | 120 | models/1c_tuned_best.joblib (44fbdd10262e) | results/1c_tuned_rbf_svm.json |

## 7. Settings of the loop

| Item | Value |
|---|---|
| settings_id | e91785ab |
| models_id | c0a27a09 |
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
| calibration_noise_rms | 0.001992 |
| threshold_rms | 0.007968 |
| room_during_run_db | 0.0 |

## 8. Level of every 0.1 s block

Each character is 4 dB above the room noise at the calibration (▁ = 0 to 4 dB, █ = 28 dB or more). The threshold is 12.0 dB (▄ and higher). `#` = word, `x` = loud sound without a word.

```
  0.0 s  ▁▁▁▁▁▁▁▁▁▁█████▁▁▁▁▁▁▇▇█▇█▇▁▁▁▁▁▁▁██████▄▁▁▁▁▁▃███
                   #####      ######       #######      ###
  5.0 s  ██▇▃▁▁▁▁▁▅███▆▄▁▁▁▁▁▁▁▁▆████▅▅▅▁▁▁▁▁▁▆▇████▇▁▁▁▁▁▁
         ###      ######        ########      #######      
 10.0 s  ▆████▄▁▁▁▁▁▂████▄▁▁▁▁▁▂████▇▇▄▁▁▁
         ######      #####      #######   
```
