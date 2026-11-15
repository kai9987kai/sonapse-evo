"""NeuroWeave command-line commands (registered next to the existing SONAPSE-EVO sub-commands).

    weave-mechanisms   list the switchable mechanisms and profiles
    weave-smoke        regression identity + every mechanism + MAX episode + package round trip
    weave-run          run episodes with any profile / mechanisms / topology / environment
    max                MAX mode: every mechanism enabled, compared with the legacy model on the same seeds
    weave-experiment   run a catalogued controlled experiment (or --list them)
    weave-topology     topology assay            weave-lesion   lesion robustness assay
    weave-export       write model package, standalone HTML runner and FPGA manifest
    weave-validate     read-only validation of a model package file
    weave-profile      measured cost versus population size
    weave-compat       check the original API surface, golden values and all-off identity
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


def _ids(text: Optional[str]) -> List[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _ints(text: Optional[str]) -> Optional[List[int]]:
    return [int(t) for t in _ids(text)] or None


def register(subparsers) -> None:
    p = subparsers.add_parser("weave-mechanisms", help="List NeuroWeave mechanisms and profiles")
    p.add_argument("--json", action="store_true")
    subparsers.add_parser("weave-smoke", help="NeuroWeave smoke verification")
    for name, helptext in (("weave-run", "Run episodes on a NeuroWeave substrate"), ("max", "MAX mode: every mechanism enabled")):
        p = subparsers.add_parser(name, help=helptext)
        p.add_argument("--profile", default="max" if name == "max" else "legacy")
        p.add_argument("--enable", default="", help="comma-separated mechanism ids to add")
        p.add_argument("--disable", default="", help="comma-separated mechanism ids to remove")
        p.add_argument("--env", default="neuroweave_1200x660" if name == "max" else "forage")
        p.add_argument("--topology", default="legacy")
        p.add_argument("--neurons", type=int, default=768)
        p.add_argument("--episodes", type=int, default=3)
        p.add_argument("--max-steps", type=int, default=0, help="override the environment's episode length")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--export", default="", help="directory for package / standalone runner / receipt")
    p = subparsers.add_parser("weave-experiment", help="Run a catalogued controlled experiment")
    p.add_argument("experiment", nargs="?", default="")
    p.add_argument("--list", action="store_true")
    p.add_argument("--full", action="store_true", help="full protocol instead of the quick smoke-sized run")
    p.add_argument("--seeds", default="", help="comma-separated seeds")
    p.add_argument("--export", default="", help="directory for the JSON receipt and CSV")
    p = subparsers.add_parser("weave-topology", help="Topology assay")
    p.add_argument("--kinds", default="")
    p.add_argument("--seeds", default="11,23,37,51")
    p.add_argument("--neurons", type=int, default=768)
    p.add_argument("--mechanisms", default="")
    p.add_argument("--export", default="")
    p = subparsers.add_parser("weave-lesion", help="Lesion robustness assay")
    p.add_argument("--severity", type=float, default=0.5)
    p.add_argument("--topology", default="multi_region")
    p.add_argument("--seeds", default="11,23,37")
    p.add_argument("--mechanisms", default="")
    p.add_argument("--recovery", type=int, default=0)
    p.add_argument("--progressive", type=int, default=0)
    p.add_argument("--export", default="")
    p = subparsers.add_parser("weave-export", help="Export model package, standalone runner and FPGA manifest")
    p.add_argument("--out", default="weave_export")
    p.add_argument("--profile", default="legacy")
    p.add_argument("--enable", default="")
    p.add_argument("--topology", default="legacy")
    p.add_argument("--neurons", type=int, default=768)
    p.add_argument("--train-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p = subparsers.add_parser("weave-validate", help="Read-only validation of a model package")
    p.add_argument("file")
    p = subparsers.add_parser("weave-profile", help="Measured cost versus population size")
    p.add_argument("--sizes", default="768,1536,3072")
    p.add_argument("--profile", default="legacy")
    p.add_argument("--steps", type=int, default=20)
    subparsers.add_parser("weave-compat", help="Check backward compatibility of the original surface")


WEAVE_COMMANDS = ("weave-mechanisms", "weave-smoke", "weave-run", "max", "weave-experiment", "weave-topology", "weave-lesion",
                  "weave-export", "weave-validate", "weave-profile", "weave-compat")


def _build(args, profile: Optional[str] = None):
    from .ecology import make_environment
    from .runtime import build_weave_substrate, canonical_motor_head

    sub = build_weave_substrate(total_neurons=args.neurons, seed=args.seed, profile=profile or args.profile, enable=_ids(getattr(args, "enable", "")),
                                disable=_ids(getattr(args, "disable", "")), topology_kind=None if args.topology == "legacy" else args.topology)
    options = {"max_steps": args.max_steps} if getattr(args, "max_steps", 0) else {}
    env_name = getattr(args, "env", "forage")
    return sub, make_environment(env_name, seed=args.seed, **options), canonical_motor_head()


def _episodes(sub, env, motor, seed: int, episodes: int):
    from .runtime import run_weave_episode

    returns, steps, wins, start = [], 0, 0, time.perf_counter()
    for k in range(episodes):
        result = run_weave_episode(env, sub, motor, seed=seed + k)
        returns.append(result.total_return)
        steps += result.steps
        wins += int(result.success)
    return returns, steps, wins, time.perf_counter() - start


def cmd_mechanisms(args) -> int:
    from .registry import NEUROWEAVE_FEATURE_IDS, PROFILES, describe_catalogue

    catalogue = describe_catalogue()
    if args.json:
        print(json.dumps({"mechanisms": catalogue, "profiles": {k: list(v) for k, v in PROFILES.items()}}, indent=2))
        return 0
    print(f"{len(catalogue)} switchable mechanisms (all disabled by default; 'legacy' is the original behaviour)\n")
    group = None
    for m in catalogue:
        if m["group"] != group:
            group = m["group"]
            print(f"[{group}]")
        origin = f"  <- NEUROWEAVE '{m['neuroweave_feature']}'" if m["neuroweave_feature"] else ""
        print(f"  {m['id']:30s} {m['kind']:8s} {m['title']}{origin}")
    print("\nProfiles: " + ", ".join(f"{k} ({len(v)})" for k, v in PROFILES.items()))
    print(f"NEUROWEAVE-LAB switches accounted for: {len(NEUROWEAVE_FEATURE_IDS)} of {len(NEUROWEAVE_FEATURE_IDS)}")
    return 0


def cmd_smoke(args) -> int:
    import torch

    from .compat import verify_legacy_identity
    from .ecology import make_environment
    from .portability import export_model_package, import_model_package, validate_model_package
    from .registry import MECHANISMS
    from .runtime import build_weave_substrate, canonical_motor_head, run_weave_episode

    t0 = time.time()
    print("=== SONAPSE-EVO NeuroWeave Smoke Verification ===")
    identity = verify_legacy_identity()
    print(f"1/5 All-off identity with the original substrate ......... {'PASS' if identity['legacy_identity'] else 'FAIL'}")
    print(f"2/5 Neutral extended path identity ....................... {'PASS' if identity['neutral_extended_path_identity'] else 'FAIL'}")
    ok = identity["legacy_identity"] and identity["neutral_extended_path_identity"]
    motor = canonical_motor_head()
    failed = []
    for mid, spec in MECHANISMS.items():
        try:
            sub = build_weave_substrate(seed=42, enable=[mid], strict=True)
            for _ in range(3):
                sub.forward_step(torch.rand(155) * 0.5)
            if not bool(torch.isfinite(sub.r).all()):
                failed.append(mid)
        except Exception as exc:
            failed.append(f"{mid} ({type(exc).__name__})")
    print(f"3/5 {len(MECHANISMS)} mechanisms attach and step individually ....... {'PASS' if not failed else 'FAIL ' + str(failed)}")
    sub = build_weave_substrate(seed=42, profile="max", strict=True)
    env = make_environment("neuroweave_1200x660", seed=42, max_steps=40)
    result = run_weave_episode(env, sub, motor, seed=42, collect_telemetry=True)
    faults = result.telemetry["faults"]
    print(f"4/5 MAX mode episode ({len(result.telemetry['enabled'])} mechanisms, {result.steps} steps) ...... {'PASS' if not faults else 'FAIL'} (return {result.total_return:.2f})")
    package = export_model_package(sub, motor_head=motor)
    report = validate_model_package(package)
    clone, _ = import_model_package(json.loads(json.dumps(package)))
    obs = torch.rand(155) * 0.5
    same = bool(torch.equal(sub.clone_substrate().forward_step(obs), clone.forward_step(obs)))
    print(f"5/5 Package export, validation and exact re-import ....... {'PASS' if report['ok'] and same else 'FAIL'}")
    ok = ok and not failed and not faults and report["ok"] and same
    print(f"\n{'All NeuroWeave smoke checks passed' if ok else 'NeuroWeave smoke FAILED'} in {time.time() - t0:.2f}s")
    return 0 if ok else 1


def cmd_run(args, max_mode: bool = False) -> int:
    from .portability import export_model_package, export_standalone_html, save_package

    sub, env, motor = _build(args)
    label = "MAX mode" if max_mode else "NeuroWeave run"
    print(f"=== SONAPSE-EVO {label} ===")
    print(f"Profile: {sub.weave.profile} | mechanisms: {len(sub.weave.enabled_ids())} | plug-ins attached: {len(sub.mechanisms)} | "
          f"neurons: {sub.total_neurons} | topology: {args.topology} | environment: {args.env}")
    returns, steps, wins, elapsed = _episodes(sub, env, motor, args.seed, args.episodes)
    telemetry = sub.weave_telemetry()
    for k, value in enumerate(returns):
        print(f"  episode {k + 1}: return {value:9.2f}")
    print(f"Mean return: {np.mean(returns):.2f} | successes: {wins}/{args.episodes} | measured {1000 * elapsed / max(1, steps):.2f} ms/step over {steps} steps")
    print(f"Mechanism faults: {len(telemetry['faults'])} | learned |dw| mean: {telemetry['edges']['mean_abs_delta']:.6f} | grown synapses: {telemetry['edges']['grown']}")
    if max_mode or sub.weave.any_enabled():
        base_sub, base_env, _ = _build(args, profile="legacy") if not (_ids(args.enable)) else (None, None, None)
        if base_sub is not None:
            b_returns, b_steps, b_wins, b_elapsed = _episodes(base_sub, base_env, motor, args.seed, args.episodes)
            print(f"Legacy model, same seeds: mean return {np.mean(b_returns):.2f} | successes {b_wins}/{args.episodes} | "
                  f"measured {1000 * b_elapsed / max(1, b_steps):.2f} ms/step")
            print(f"Paired difference (this run minus legacy): {np.mean(returns) - np.mean(b_returns):+.2f} over {args.episodes} episode(s); "
                  "descriptive only - use weave-experiment for a controlled comparison")
        sig = telemetry["signals"]
        print("Affect: " + ", ".join(f"{k} {v:+.3f}" for k, v in sig["affect"].items()))
        print("Modulators: " + ", ".join(f"{k} {v:+.3f}" for k, v in sig["modulators"].items()))
    if args.export:
        out = Path(args.export)
        package = export_model_package(sub, motor_head=motor, environment=env if hasattr(env, "state_dict") else None,
                                       training_history={"episodes": args.episodes, "returns": [float(r) for r in returns]})
        size = save_package(package, str(out / "model_package.json"))
        export_standalone_html(package, str(out / "standalone.html"))
        print(f"Exported model package ({size / 1e6:.2f} MB) and standalone runner to {out}")
    return 0 if not telemetry["faults"] else 1


def cmd_experiment(args) -> int:
    from .experiments import export_receipt, list_experiments, run_catalog_experiment

    if args.list or not args.experiment:
        for e in list_experiments():
            print(f"{e['id']:34s} [{e['kind']:8s}] {e['title']}")
        return 0
    receipt = run_catalog_experiment(args.experiment, seeds=_ints(args.seeds), quick=not args.full)
    print(f"=== Experiment: {receipt['title']} ===")
    summary = receipt["results"]["summary"]
    for arm, s in summary.items():
        if "heldout_mean" in s:
            cost = f" | {s['measured_ms_per_step']:.2f} ms/step" if "measured_ms_per_step" in s else ""
            print(f"  {arm:40s} held-out return {s['heldout_mean']:9.3f} +/- {s.get('heldout_std', 0.0):.3f}{cost}")
        else:
            print(f"  {arm:40s} " + ", ".join(f"{k} {v:.3f}" for k, v in list(s.items())[:5]))
    for name, cmp in receipt["results"]["comparisons"].items():
        if cmp.get("mean_diff") is not None:
            print(f"  {name}: mean paired difference {cmp['mean_diff']:+.4f} over {cmp['n_seeds']} seeds -> {cmp['verdict']} ({cmp['reason']})")
    print(f"Verdict: {receipt['verdict']} | evidence tier: {receipt['evidence_tier']} | digest {receipt['result_digest'][:16]}")
    if args.export:
        if "runs" in receipt["results"] and isinstance(receipt["results"]["runs"], dict) and receipt.get("kind") == "paired":
            print("Wrote " + ", ".join(export_receipt(receipt, args.export).values()))
        else:
            Path(args.export).mkdir(parents=True, exist_ok=True)
            path = Path(args.export) / f"{receipt['experiment_id']}.json"
            path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
            print(f"Wrote {path}")
    return 0


def _write(receipt, directory: str) -> None:
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"{receipt['experiment_id']}.json"
        path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {path}")


def cmd_topology(args) -> int:
    from .experiments import run_topology_assay

    kwargs = {"kinds": tuple(_ids(args.kinds))} if args.kinds else {}
    receipt = run_topology_assay(seeds=tuple(_ints(args.seeds)), total_neurons=args.neurons, mechanisms=tuple(_ids(args.mechanisms)), **kwargs)
    print(f"=== Topology assay ({args.neurons} neurons, seeds {receipt['seeds']}) ===")
    print(f"{'topology':24s} {'return':>9s} {'success':>8s} {'steps':>6s} {'edges':>7s} {'cluster':>8s} {'modular':>8s} {'path':>6s} {'lesion':>7s} {'ms/step':>8s}")
    for kind in receipt["results"]["ranking_by_heldout_return"]:
        s = receipt["results"]["summary"][kind]
        print(f"{kind:24s} {s['heldout_mean']:9.2f} {s['success_rate']:8.2f} {s['mean_steps']:6.1f} {s['edges_mean']:7.0f} {s['clustering_coefficient_mean']:8.3f} "
              f"{s['modularity_mean']:8.3f} {s['characteristic_path_length_mean']:6.2f} {s['lesion_retention_mean']:7.2f} {s['measured_ms_per_step']:8.2f}")
    print("Evidence tier: " + receipt["evidence_tier"] + " | a ranking on this synthetic task is not a claim about biological wiring")
    _write(receipt, args.export)
    return 0


def cmd_lesion(args) -> int:
    from .experiments import run_lesion_assay

    receipt = run_lesion_assay(seeds=tuple(_ints(args.seeds)), severity=args.severity, topology_kind=args.topology,
                               mechanisms=tuple(_ids(args.mechanisms)), recovery_episodes=args.recovery, progressive_steps=args.progressive)
    print(f"=== Lesion robustness assay ({args.topology}, severity {args.severity}) ===")
    print(f"{'lesion':30s} {'removed':>8s} {'intact':>8s} {'targeted':>9s} {'random':>8s}   success: intact / targeted / random")
    for kind, s in receipt["results"]["summary"].items():
        print(f"{kind:30s} {s['edges_removed_mean']:8.0f} {s['intact_mean']:8.2f} {s['targeted_mean']:9.2f} {s['matched_random_mean']:8.2f}   "
              f"{s['intact_success_rate']:.2f} / {s['targeted_success_rate']:.2f} / {s['random_success_rate']:.2f}")
    print(receipt["interpretation_note"])
    _write(receipt, args.export)
    return 0


def cmd_export(args) -> int:
    from .hardware import build_fpga_manifest, validate_fpga_manifest
    from .portability import export_model_package, export_standalone_html, save_package, validate_model_package

    args.env, args.max_steps, args.disable = "forage", 0, ""
    sub, env, motor = _build(args)
    returns, _, _, _ = _episodes(sub, env, motor, args.seed, args.train_episodes) if args.train_episodes > 0 else ([], 0, 0, 0.0)
    out = Path(args.out)
    package = export_model_package(sub, motor_head=motor, training_history={"episodes": args.train_episodes, "returns": [float(r) for r in returns]})
    size = save_package(package, str(out / "model_package.json"))
    export_standalone_html(package, str(out / "standalone.html"))
    manifest = build_fpga_manifest(sub)
    (out / "fpga_mapping.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    report = validate_model_package(package)
    print(f"model_package.json  {size / 1e6:.2f} MB  valid={report['ok']}  digest {package['integrity']['digest'][:16]}")
    print("standalone.html     frozen inference runner (lists what it does not execute)")
    print(f"fpga_mapping.json   {manifest['synapses']} int8 synapses, {manifest['tileSchedule']['tiles']} tiles, issues={validate_fpga_manifest(manifest)} (mapping manifest only)")
    print(f"Written to {out.resolve()}")
    return 0 if report["ok"] else 1


def cmd_validate(args) -> int:
    from .portability import LIMITS, inspect_model_package

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    if path.stat().st_size > LIMITS["max_bytes"]:
        print(f"REJECTED: file is larger than {LIMITS['max_bytes']} bytes")
        return 1
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print("REJECTED: the file is not UTF-8 text")
        return 1
    info = inspect_model_package(text)
    v = info["validation"]
    print(f"{'VALID' if v['ok'] else 'REJECTED'}  format {info['format']} {info['formatVersion']}  counts {info['counts']}")
    for issue in v["issues"]:
        print(f"  issue:   {issue}")
    for warning in v["warnings"]:
        print(f"  warning: {warning}")
    if v["ok"]:
        caps = {k: val for k, val in (info["capabilities"] or {}).items() if k != "notes"}
        print(f"  mechanisms enabled: {len(info['mechanisms'])} | capabilities: {caps}")
    print("  (read-only: nothing was imported)")
    return 0 if v["ok"] else 1


def cmd_profile(args) -> int:
    from .hardware import scaling_profile

    print(f"=== Measured cost versus population size (profile '{args.profile}') ===")
    print(f"{'neurons':>8s} {'edges':>8s} {'ms/step':>9s} {'steps/s':>9s} {'state MB':>9s}")
    for row in scaling_profile(sizes=_ints(args.sizes), profile=args.profile, steps=args.steps):
        m, c = row["measured"], row["config"]
        print(f"{c['neurons']:8d} {c['edges']:8d} {m['ms_per_step']:9.3f} {m['steps_per_second']:9.1f} {m['total_bytes'] / 1e6:9.2f}")
    print("Wall-clock and bytes are measured on this machine; no acceleration is claimed.")
    return 0


def cmd_compat(args) -> int:
    from .compat import check_compatibility, golden_baseline, verify_legacy_identity

    root = Path(__file__).resolve().parents[3] / "tests" / "weave_baseline"
    if not root.exists():
        print(f"Baseline directory not found: {root}")
        return 1
    manifest = json.loads((root / "compatibility_manifest.json").read_text(encoding="utf-8"))
    golden = json.loads((root / "golden.json").read_text(encoding="utf-8"))
    violations = check_compatibility(manifest)
    live = golden_baseline()
    drift = [k for k in live if json.dumps(live[k], sort_keys=True) != json.dumps(golden[k], sort_keys=True)]
    identity = verify_legacy_identity()
    n_methods = sum(len(c["methods"]) for m in manifest["modules"].values() for c in m["classes"].values())
    print("=== SONAPSE-EVO backward-compatibility check ===")
    print(f"API surface ({len(manifest['modules'])} modules, {n_methods} methods, {len(manifest['cli']['subcommands'])} CLI commands, "
          f"{len(manifest['dom_ids'])} dashboard hooks): {'PASS' if not violations else 'FAIL'}")
    for v in violations[:20]:
        print(f"  {v}")
    print(f"Golden numeric baseline: {'PASS' if not drift else 'FAIL ' + str(drift)}")
    print(f"All-off identity: {'PASS' if identity['legacy_identity'] else 'FAIL'} | neutral extended path: "
          f"{'PASS' if identity['neutral_extended_path_identity'] else 'FAIL'}")
    ok = not violations and not drift and identity["legacy_identity"] and identity["neutral_extended_path_identity"]
    return 0 if ok else 1


def dispatch(args) -> Optional[int]:
    """Return an exit code when ``args`` is a NeuroWeave command, else ``None``."""
    table = {"weave-mechanisms": cmd_mechanisms, "weave-smoke": cmd_smoke, "weave-run": cmd_run, "weave-experiment": cmd_experiment,
             "weave-topology": cmd_topology, "weave-lesion": cmd_lesion, "weave-export": cmd_export, "weave-validate": cmd_validate,
             "weave-profile": cmd_profile, "weave-compat": cmd_compat}
    if args.subcommand == "max":
        return cmd_run(args, max_mode=True)
    if args.subcommand in table:
        return table[args.subcommand](args)
    return None


__all__ = ["register", "dispatch", "WEAVE_COMMANDS"]
