# Data provenance — v0.6.0

## UCI Heart Disease

Official source: UCI Machine Learning Repository, Heart Disease dataset.
Four processed databases were used: Cleveland, Hungary, Switzerland, and
VA Long Beach. Raw source files are not redistributed in this package.

SHA-256 of the files used for the executed validation:

- processed.cleveland.data — `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
- processed.hungarian.data — `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796`
- processed.switzerland.data — `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314`
- processed.va.data — `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f`

## NHANES 2015–2016 diabetes experiment

Original documentation/source: CDC/NCHS NHANES 2015–2016. The execution
used CSV translations of the public NHANES SAS transport files from the
`protobi/nhanes-continuous` public repository because direct binary XPT
retrieval was unavailable in the execution environment. The repository states
that these CSV files are translations of the NHANES XPT tables. Raw files are
not redistributed in this package.

Tables used:

- `GHB_I` — HbA1c (`LBXGH`)
- `GLU_I` — fasting plasma glucose (`LBXGLU`)
- `OGTT_I` — 2-hour OGTT glucose (`LBXGLT`)
- `DEMO_I` — demographics and survey design variables
- `BMX_I` — BMI and waist circumference
- `DIQ_I` — self-reported diabetes status

SHA-256 of the CSV files used:

- GHB_I.csv — `40ad988fbd2f12db9d06d8e2384c14940c0cb4897d76099a68b988c165f87dac`
- GLU_I.csv — `927e05678c0ee8040c08bb95df61139bd00a06c57b26144df2e1542f72efc3e5`
- OGTT_I.csv — `8409905fe107e1bb1dd4f5fc4faf96c1266dbd9f25f2f7e68890ca658351f068`
- DEMO_I.csv — `47dbc233abda14f0adb6fd3eeb73f66e7ad4cd26dfa692c56f270c4b27283e4c`
- BMX_I.csv — `bdef8797c4061daf3b50380bfbaec763a880c8832817316a9bea6755b3a851de`
- DIQ_I.csv — `bd1f4bafb84cc50430e83e39fb550b05996196fad9c363705adcbfb335879d49`

The benchmark does not use HbA1c, FPG, or OGTT values as predictor features.
They are used only to construct/evaluate the documented laboratory-definition
labels and the explicitly declared research canonical endpoint.
