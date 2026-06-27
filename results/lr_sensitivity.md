# Learning Rate Sensitivity Results

## Summary

| Learning Rate | Initial Loss | Final Loss | Behavior |
|--------------|-------------|------------|----------|
| 1e-1 | NaN | NaN | Gradient explosion |
| 1e-2 | 19.654 | 19.540 | Oscillation and divergence |
| 1e-3 | 19.304 | 7.250 | High initial damage, slow recovery |
| 1e-4 | 1.334 | 0.964 | Stable convergence ✅ |
| 1e-5 | 1.834 | 0.993 | Slightly slower, comparable result |
| 1e-6 | 2.242 | 1.481 | Too slow, underfitting |

## Detailed Loss Curves

### lr=1e-4 (Baseline)
| Iter | Loss |
|------|------|
| 10 | 1.334 |
| 20 | 1.079 |
| 30 | 1.127 |
| 40 | 0.915 |
| 50 | 0.996 |
| 60 | 1.002 |
| 70 | 1.000 |
| 80 | 0.937 |
| 90 | 0.983 |
| 100 | 0.964 |

### lr=1e-5
| Iter | Loss |
|------|------|
| 10 | 1.834 |
| 20 | 1.358 |
| 30 | 1.237 |
| 40 | 0.976 |
| 50 | 1.067 |
| 60 | 1.058 |
| 70 | 1.054 |
| 80 | 0.967 |
| 90 | 1.029 |
| 100 | 0.993 |

### lr=1e-6
| Iter | Loss |
|------|------|
| 10 | 2.242 |
| 20 | 2.126 |
| 30 | 1.920 |
| 40 | 1.595 |
| 50 | 1.745 |
| 60 | 1.609 |
| 70 | 1.615 |
| 80 | 1.539 |
| 90 | 1.565 |
| 100 | 1.481 |

### lr=1e-3 (High initial damage)
| Iter | Loss |
|------|------|
| 10 | 19.304 |
| 20 | 15.745 |
| 30 | 10.422 |
| 40 | 9.111 |
| 50 | 8.557 |
| 60 | 8.331 |
| 70 | 7.882 |
| 80 | 7.351 |
| 90 | 7.609 |
| 100 | 7.250 |

### lr=1e-2 (Divergence)
| Iter | Loss |
|------|------|
| 10 | 19.654 |
| 20 | 13.009 |
| 30 | 19.643 |
| 40 | 22.930 |
| 50 | 21.798 |
| 60 | 19.228 |
| 70 | 19.613 |
| 80 | 20.044 |
| 90 | 19.308 |
| 100 | 19.540 |

### lr=1e-1 (Gradient explosion)
| Iter | Loss |
|------|------|
| 10 | NaN |
| 20 | NaN |
| 30 | NaN |
| 40 | NaN |
| 50 | NaN |
