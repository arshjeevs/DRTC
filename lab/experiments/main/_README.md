The four conditions of the locked scope (lab/PLAN.md, "LOCKED SCOPE", 2026-11-20). The same on every task; the real
network (laptop -> GPU server over WiFi, raw images), no simulated latency.

| file | condition |
|---|---|
| baseline.yaml | 1: upstream LeRobot as it ships (no DRTC, no filter, gives up after 1 s without actions) |
| baseline_faults.yaml | 2: as 1 + the fault mix |
| drtc_faults.yaml | 3: DRTC (our version), no filter, holds through stalls, + the fault mix |
| drtc_smooth_faults.yaml | 4: as 3 + the Butterworth filter + adaptive playback (method A) |

The fault mix (times from the trial start): latency spike +1 s at 2 s; every reply dropped 4-5 s; full outage (both
directions) 6-7.5 s.
