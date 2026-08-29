"""Fidelity-feedback calibration of generation distributions.

`controller` holds the pure control law; `artifact` holds the on-disk format.
Neither imports a generator or makes an API call — the driver in
`framework.calibrate` owns everything that talks to a provider.
"""
