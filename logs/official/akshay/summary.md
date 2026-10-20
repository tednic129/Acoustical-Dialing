# Demo logs: summary

Updated 2026-10-12 00:07:27. 9 run(s) in this folder.

A run counts when the loop heard at least one word. A run that you stop early, or in which a word is missed, counts as a failed run. A run that ended with an error does not count. Move the logs of dry runs to another folder before you take the counters for the report.

## Pizza-test counters (correct numbers / counted runs)

| Language | Source | Voice | Runs | 1.a default | 1.a tuned | 1.b default | 1.b tuned | 1.c default | 1.c tuned |
|---|---|---|---|---|---|---|---|---|---|
| German | microphone | training voice | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 |
| German | recordings (simulation) | new voice | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| German | recordings (simulation) | training voice | 1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| English | microphone | training voice | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| English | recordings (simulation) | new voice | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| English | recordings (simulation) | training voice | 1 | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## All runs

| Run | Lang. | Source | Voice | Target | Words | Status | Settings ID / models ID | 1.a default | 1.a tuned | 1.b default | 1.b tuned | 1.c default | 1.c tuned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [2026-10-11_233432_de_simulation_training](2026-10-11_233432_de_simulation_training.md) | de | simulation | training | `0631123456` | 10/10 | complete | e91785ab / c0a27a09 | pass | FAIL | pass | pass | pass | pass |
| [2026-10-11_234901_de_simulation_new](2026-10-11_234901_de_simulation_new.md) | de | simulation | new | `0631123456` | 10/10 | complete | e91785ab / c0a27a09 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| [2026-10-11_235035_en_simulation_training](2026-10-11_235035_en_simulation_training.md) | en | simulation | training | `0631123456` | 10/10 | complete | e91785ab / 73a1c2a1 | pass | FAIL | pass | pass | pass | pass |
| [2026-10-11_235134_en_simulation_new](2026-10-11_235134_en_simulation_new.md) | en | simulation | new | `0631123456` | 10/10 | complete | e91785ab / 73a1c2a1 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| [2026-10-11_235814_en_microphone_training](2026-10-11_235814_en_microphone_training.md) | en | microphone | training | `0631123456` | 0/10 | stopped | e91785ab / 73a1c2a1 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| [2026-10-11_235821_en_microphone_training](2026-10-11_235821_en_microphone_training.md) | en | microphone | training | `0631123456` | 10/10 | complete | e91785ab / 73a1c2a1 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| [2026-10-12_000047_en_microphone_training](2026-10-12_000047_en_microphone_training.md) | en | microphone | training | `0631123456` | 10/10 | complete | e91785ab / 73a1c2a1 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| [2026-10-12_000420_de_microphone_training](2026-10-12_000420_de_microphone_training.md) | de | microphone | training | `0631123456` | 10/10 | complete | e91785ab / c0a27a09 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| [2026-10-12_000712_de_microphone_training](2026-10-12_000712_de_microphone_training.md) | de | microphone | training | `0631123456` | 10/10 | complete | e91785ab / c0a27a09 | FAIL | FAIL | FAIL | FAIL | FAIL | pass |

Settings ID: loop settings and event-extraction settings. Models ID: the model files of the language. A change from the previous run of the same language means that the runs do not test the same system.
