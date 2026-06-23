"""The CMIP-DECK CO2 schedule (experiments/run_dino_deck.py:co2_for_experiment).

Network-free, JAX-free unit checks of the forcing schedule so the DECK driver's
science (the CO2 pathway each experiment imposes) is testable without a control
checkpoint or the cached datasets.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "run_dino_deck.py",
)
_spec = importlib.util.spec_from_file_location("run_dino_deck", _PATH)
deck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deck)

C0 = 280.0


def test_picontrol_is_flat():
    for yr in (0.0, 50.0, 150.0):
        assert deck.co2_for_experiment("piControl", yr, C0) == C0


def test_abrupt_levels():
    assert deck.co2_for_experiment("abrupt-2xCO2", 0.0, C0) == 2 * C0 == 560.0
    assert deck.co2_for_experiment("abrupt-4xCO2", 0.0, C0) == 4 * C0 == 1120.0
    # abrupt = step change, constant thereafter
    assert deck.co2_for_experiment("abrupt-4xCO2", 99.0, C0) == 1120.0


def test_1pct_ramp_doubles_and_quadruples():
    # compound 1%/yr: 2x near year 70 (1.01**70 ~ 2.007), 4x near year 140
    assert deck.co2_for_experiment("1pctCO2", 0.0, C0) == pytest.approx(C0)
    assert deck.co2_for_experiment("1pctCO2", 70.0, C0) == pytest.approx(
        C0 * 1.01**70, rel=1e-9
    )
    assert deck.co2_for_experiment("1pctCO2", 70.0, C0) == pytest.approx(
        560.0, rel=0.01
    )
    # monotonic increasing before the cap
    assert deck.co2_for_experiment("1pctCO2", 50.0, C0) > deck.co2_for_experiment(
        "1pctCO2", 49.0, C0
    )


def test_1pct_capped_at_4x():
    # well past quadrupling the ramp is capped at 4*C0 (does not run away)
    assert deck.co2_for_experiment("1pctCO2", 200.0, C0) == 4 * C0
    assert deck.co2_for_experiment("1pctCO2", 1000.0, C0) == 4 * C0


def test_unknown_experiment_raises():
    with pytest.raises(ValueError):
        deck.co2_for_experiment("ssp585", 0.0, C0)


if __name__ == "__main__":
    test_picontrol_is_flat()
    test_abrupt_levels()
    test_1pct_ramp_doubles_and_quadruples()
    test_1pct_capped_at_4x()
    test_unknown_experiment_raises()
    print("all DECK schedule tests passed")
