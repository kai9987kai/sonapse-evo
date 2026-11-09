import pytest
import time

from sonapse_evo.observatory.live_state import LiveObservatoryState


def test_live_observatory_state_lifecycle():
    state = LiveObservatoryState()
    assert state.step_idx == 0
    assert len(state.telemetry_history) > 0

    # Step simulation
    state.tick_simulation()
    assert state.step_idx == 1

    # Check full state schema
    full = state.get_full_state()
    assert "mode_a" in full
    assert "mode_b" in full
    assert "connectome" in full
    assert "telemetry_stream" in full
    assert full["connectome"]["total_neurons"] == 768
    assert len(full["mode_b"]["pareto_points"]) > 0

    # Test Mode B live discovery advance
    gen_before = state.current_generation
    disc_res = state.trigger_discovery_step()
    assert disc_res["generation"] == gen_before + 1
    assert state.current_generation == gen_before + 1
    assert len(state.population_data) > 0

    # Test Mode A lesion event
    les_res = state.trigger_lesion_event()
    assert les_res["status"] == "LESION_ACTIVE"
    assert state.mode_a_status == "LESION_ACTIVE"
    assert state.deficit_score > 0.50

    # Test Mode A repair event
    rep_res = state.trigger_repair_event()
    assert rep_res["status"] == "COMMITTED"
    assert state.mode_a_status == "COMMITTED"
    assert rep_res["recovery_ratio"] > 1.0
    assert len(state.transactions) > 1
