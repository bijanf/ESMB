# Mid-Holocene (6 ka) paleo experiment — P5

The fourth working-ESM non-negotiable: a **paleo configuration** that responds to altered
boundary conditions. We run the classic **PMIP mid-Holocene (6 ka)** orbital experiment — the
cleanest paleo case for a coupled model because it changes *only* the orbit (no ice sheets /
geography), so the response is an unambiguous test of the model's forced sensitivity.

## Forcing

Earth's orbit 6 000 yr ago differed from today in the three Milankovitch parameters
(`chronos_esm/orbital.py`, PMIP4 values):

| | obliquity | eccentricity | longitude of perihelion |
|---|---|---|---|
| **PI (1850)** | 23.459° | 0.016764 | 280.33° |
| **6 ka** | 24.105° | 0.018682 | 180.87° |

The net effect is a **seasonal redistribution** of insolation: with perihelion in boreal
summer and a larger tilt, 6 ka receives **+20–30 W/m² more NH summer (JJA) insolation** (and
less in NH winter); the annual-global mean is unchanged to <0.5 W/m². The forcing is validated
against textbook / PMIP4 numbers in `tests/test_orbital.py`, and it is differentiable in every
orbital parameter (`d(insolation)/d(obliquity)` etc.).

## Design (PMIP-style)

The active dino model is given a **real seasonal cycle** (`DinoCoupledModel(seasonal=True,
orbit=...)`, see [seasonal cycle](#seasonal-cycle) below). Both runs branch from the **same**
seasonal-control checkpoint and use the **same q-flux** (the climatological flux correction,
held fixed across the perturbation); only the orbit differs, so the 6 ka − PI difference
isolates the orbital response. Each run is 40 yr (free q-flux ocean, weak anomaly restoring);
the JJA/DJF climatology is accumulated over the back 25 yr.

```bash
CK=/p/tmp/$USER/seasonal_control_pi2/state_d014600            # seasonal-control checkpoint + q-flux
python experiments/run_paleo_midholocene.py --ckpt $CK --orbit pi  --years 40 --outdir <dir>
python experiments/run_paleo_midholocene.py --ckpt $CK --orbit 6ka --years 40 --outdir <dir>
python experiments/plot_paleo_midholocene.py --pi <dir>/clim_pi.npz --ho <dir>/clim_6ka.npz \
    --out docs/figures/paleo_midholocene.pdf
```

## Result (`docs/figures/paleo_midholocene.pdf`)

The model reproduces the **correct large-scale mid-Holocene fingerprint**:

**NH summer warming (robust)** — the direct insolation response, the clearest signal:

| region | JJA 2 m-T, 6 ka − PI |
|---|---|
| NH 20–60°N | **+1.1 K** |
| Arctic 60–90°N | **+1.9 K** (Arctic-amplified) |
| SH 60–90°S | −0.1 K (slight cooling) |
| global (JJA) | +0.17 K (annual-global ≈ 0, as orbital forcing requires) |

**Monsoon intensification + northward ITCZ** (JJA precip, 6 ka − PI):

| sector | PI → 6 ka | change |
|---|---|---|
| S/SE Asia (10–30°N, 60–120°E) | 0.64 → 0.83 mm/day | **+31 %** |
| N. America (10–30°N) | 2.36 → 2.84 mm/day | **+20 %** |
| tropical ITCZ latitude | +5.94° → +6.15° | **+0.2° north** |

So the model **warms the NH summer and strengthens the monsoons it resolves**, in the right
direction — a genuine, differentiable orbital response.

## Honest limitations

* **No "Green Sahara".** N. Africa/Sahel stays dry (0.15 → 0.10 mm/day) instead of greening.
  The model's African monsoon is essentially absent (~0.1 mm/day) to begin with, so there is
  no monsoon there to enhance — the known T31 **weak-ITCZ** limitation (see the README
  validation notes), shared with many coarse models that also under-predict the Green Sahara.
* **Modest magnitude.** The tropical-precip and ITCZ-shift responses are weaker than typical
  PMIP models, again traceable to the coarse, SST-slaved single-humidity atmosphere. The
  *thermal* response (NH summer warming) is the more quantitatively trustworthy signal.

## Seasonal cycle

The active dino coupling was **perpetual-equinox** (insolation hardcoded at `day_of_year=80`).
`DinoCoupledModel(seasonal=True, orbit=...)` now recomputes insolation each interval from the
model day via the orbital forcing (`_insolation`; calendar→solar-longitude by Kepler), routed
to the ocean / land / sea-ice surface SW. Gated behind `seasonal` (default off → bit-identical
legacy behaviour). The seasonal surface SW reproduces `compute_solar_insolation`'s convention
exactly (`TOA·(1−albedo)·0.60`); a raw-TOA override double-counts albedo and ~2× the ocean SW
(caught in calibration — see `tests/test_seasonal_cycle.py`). Validated:
`tests/test_orbital.py` (6), `tests/test_seasonal_cycle.py` (3).
