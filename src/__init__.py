"""
ClimateForte Solar DSM MVP — Source Package

Pipeline Stages:
  S2: src.physics_baseline   — PVLib Physics Engine
  S3: src.ml_correction      — XGBoost Residual Inference
  S4: src.dsm_settlement     — DSM Penalty Calculator + Reporting
"""
