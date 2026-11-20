# Mobile Health Sensor Fusion

Synthetic on-device health-event classifier combining motion, audio-summary,
and environmental sensor features. It compares complete-data training with
modality-dropout augmentation under missing-sensor conditions and reports
AUROC by missing modality. No personal data are used.

Run: `python -m mobile_health.experiment --output outputs/demo`
