# Data Dictionary — Industrial Pump Fleet Sensor Data

> Read this before modelling. Column names in train.csv use generic labels (sensor_1 to sensor_20).
> The physical interpretation of each channel is provided here.

## Unit Files

### train_units.csv
| Column | Type | Description |
|---|---|---|
| unit_id | string | Unique pump identifier |
| commissioning_date | date | When pump entered service |
| cycles_recorded | int | Total operating cycles in the dataset |
| event_flag | int | 1 = ran to failure; 0 = monitoring stopped (censored) |

### train.csv / sample_test_input.csv
| Column | Type | Description |
|---|---|---|
| unit_id | string | Pump identifier (matches train_units) |
| cycle | int | Operating cycle number (1 = first hour of service) |
| sensor_1–20 | float | Sensor readings (see table below) |

## Sensor Channel Reference

| Column | Physical Channel | Units | Notes |
|---|---|---|---|
| sensor_1 | Fan inlet temperature | °R (Rankine) | Operational setpoint — varies by duty |
| sensor_2 | LPC outlet temperature | °R | Condition indicator |
| sensor_3 | HPC outlet temperature | °R | Condition indicator — rises with degradation |
| sensor_4 | LPT outlet temperature | °R | Condition indicator |
| sensor_5 | Fan inlet pressure | psia | Operational — near constant |
| sensor_6 | Bypass-duct pressure | psia | Operational |
| sensor_7 | HPC outlet pressure | psia | Key degradation channel |
| sensor_8 | Physical fan speed | rpm | Condition indicator |
| sensor_9 | Physical core speed | rpm | Condition indicator |
| sensor_10 | Engine pressure ratio | — | Derived; low variance |
| sensor_11 | HPC outlet static pressure | psia | Condition indicator |
| sensor_12 | Ratio of fuel flow to Ps30 | pps/psia | Key efficiency channel |
| sensor_13 | Corrected fan speed | rpm | Important degradation signal |
| sensor_14 | Corrected core speed | rpm | Important degradation signal |
| sensor_15 | Bypass ratio | — | Low variance setpoint |
| sensor_16 | Burner fuel-air ratio | — | Often near-constant |
| sensor_17 | Bleed enthalpy | — | Low variance |
| sensor_18 | Required fan speed | rpm | Setpoint — not a condition indicator |
| sensor_19 | Required fan conversion speed | rpm | Setpoint |
| sensor_20 | High-pressure turbine cool air flow | lbm/s | Condition channel |

> Note: Some channels may be flat (near-zero variance) for specific pump models —
> the signal quality classification in Notebook 01 will identify these automatically.
> Do not use dead channels as model features.
