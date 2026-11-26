"""Online cooperative/zero-sum PPO training and locked held-out evaluation.

No generator weight updates. Every step obtains fresh model proposals conditioned
on the current protein/ligand state. Test trajectories never provide training reward.
"""

import argparse, copy, json, os, random, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from .common import (
    ROOT,
    CODE_ROOT,
    atomic_json,
    append_json,
    identity,
    sha,
    seed_all,
    ligand,
)
from .models import PPO
from .features import encode, chemical_mask
from .proposals import Proposals
from .evaluator import Evaluator


class Campaign:
    def __init__(self, args):
        self.args = args
        self.cfg = json.loads(args.config.read_text())
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(args.manifest.read_text())
        if self.manifest["stage"] != "complete" or self.manifest[
            "config_sha256"
        ] != sha(args.config):
            raise ValueError("Unprepared or mismatched dataset")
        seed_all(args.seed)
        self.ppo = PPO(self.cfg["training"])
        self.oracle = Evaluator(self.out / "oracle", self.cfg)
        self.proposals = Proposals(args.config, self.cfg, self.out)
        self.step = 0
        self.episode = 0
        self.best_validation = -float("inf")
        self.last_validation = None
        self.initialized = False
        self.training_successes = 0
        self.status = {
            "stage": "initializing",
            "setting": args.setting,
            "seed": args.seed,
            "config": self.cfg,
            "config_sha256": sha(args.config),
            "manifest_sha256": sha(args.manifest),
            "generator_training": False,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "source_root": str(CODE_ROOT),
            "source_manifest_sha256": (
                sha(CODE_ROOT / "source_hashes.json")
                if (CODE_ROOT / "source_hashes.json").exists()
                else None
            ),
        }
        self.rng = np.random.default_rng(args.seed)
        seed_all(args.seed)
        self.resume()
        self.save_status()

    def save_status(self, **kw):
        self.status.update(
            kw,
            episodes_completed=self.episode,
            training_successes=self.training_successes,
            oracle_requests=self.oracle.calls,
            oracle_cache_hits=self.oracle.cache_hits,
            oracle_errors=self.oracle.errors,
            ligands_generated=self.proposals.generated,
            valid_generated=self.proposals.valid,
            pocket_proposals=self.proposals.pocket_generated,
        )
        atomic_json(self.out / "status.json", self.status)

    def checkpoint(self):
        payload = {
            "agents": self.ppo.agents.state_dict(),
            "optimizer": self.ppo.optim.state_dict(),
            "buffer": self.ppo.buffer,
            "episode": self.episode,
            "step": self.step,
            "best_validation": self.best_validation,
            "last_validation": self.last_validation,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": self.rng.bit_generator.state,
            "python_rng": random.getstate(),
            "proposal_counter": self.proposals.counter,
            "training_successes": self.training_successes,
            "config_sha256": sha(self.args.config),
            "manifest_sha256": sha(self.args.manifest),
            "counters": {
                k: getattr(self.proposals, k)
                for k in ["generated", "valid", "pocket_generated", "failures"]
            },
            "oracle_counters": {
                k: getattr(self.oracle, k) for k in ["calls", "cache_hits", "errors"]
            },
        }
        tmp = self.out / "checkpoint.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.out / "checkpoint.pt")

    def resume(self):
        path = self.out / "checkpoint.pt"
        if not path.exists():
            return
        c = torch.load(path, map_location="cpu", weights_only=False)
        if c["config_sha256"] != sha(self.args.config) or c["manifest_sha256"] != sha(
            self.args.manifest
        ):
            raise ValueError("Resume config/data mismatch")
        self.ppo.agents.load_state_dict(c["agents"])
        self.ppo.optim.load_state_dict(c["optimizer"])
        self.ppo.buffer = c["buffer"]
        self.episode = c["episode"]
        self.step = c["step"]
        self.best_validation = c["best_validation"]
        self.last_validation = c["last_validation"]
        self.training_successes = c["training_successes"]
        torch.set_rng_state(c["torch_rng"])
        self.rng.bit_generator.state = c["numpy_rng"]
        random.setstate(c["python_rng"])
        # Never overwrite partial raw artifacts from an interrupted uncheckpointed episode.
        existing = [
            int(p.name) for p in (self.out / "proposals").glob("*") if p.name.isdigit()
        ]
        self.proposals.counter = max(
            [c["proposal_counter"]] + [i + 1 for i in existing]
        )
        for k, v in c["counters"].items():
            setattr(self.proposals, k, max(getattr(self.proposals, k), v))

    def ensemble(self, target):
        path = ROOT / "data/full_v1/ensembles" / target["id"]
        manifest = path / "ensemble.json"
        if not manifest.exists():
            env = dict(
                os.environ,
                PYTHONPATH=str(CODE_ROOT),
                OMP_NUM_THREADS="4",
                OPENMM_CPU_THREADS="4",
            )
            with (self.out / "ensemble.log").open("a") as log:
                subprocess.run(
                    [
                        str(ROOT / "envs/flowr/bin/python"),
                        "-u",
                        "-m",
                        "research.ensemble",
                        "--protein",
                        target["protein"],
                        "--output",
                        str(path),
                        "--config",
                        str(self.args.config),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=21600,
                )
        e = json.loads(manifest.read_text())
        train = [target["protein"]] + [
            x["path"] for x in e["rows"] if x["split"] == "train" and x["eligible"]
        ]
        # Native + evenly spaced frames, identical across methods and policy seeds.
        ids = np.linspace(0, len(train) - 1, min(8, len(train)), dtype=int)
        train = [train[i] for i in ids]
        holdout = [
            x["path"] for x in e["rows"] if x["split"] == "heldout" and x["eligible"]
        ]
        return train, holdout

    def initial(self, target):
        result = self.oracle.evaluate(
            target["protein"], target["ligand"], target, seed=self.args.seed
        )
        return float(result["utility"]) if result["feasible"] else -20.0, result

    def episode_train(self, target):
        h = self.cfg["training"]["horizon"]
        p, l = target["protein"], target["ligand"]
        u, initial = self.initial(target)
        records = [[], []]
        rewards = [[], []]
        trace = []
        if self.args.setting == "competitive":
            states, _ = self.ensemble(target)
        for k in range(h):
            seed = self.args.seed + 100003 * self.episode + k
            remaining = (h - k) / h
            genp = (
                p
                if self.args.setting == "cooperative"
                else states[(self.episode + k) % len(states)]
            )
            ligs, pockets = self.proposals.generate(
                genp, l, target, seed, cooperative=self.args.setting == "cooperative"
            )
            mask = chemical_mask(ligs, target["ligand"], self.cfg["constraints"])
            mask[0] = True
            if len(ligs) == 1:
                trace.append(
                    {"step": k, "error": "No valid generated molecules", "reward": -2.0}
                )
                break
            s = encode(p, l, u, remaining)
            lf = np.array([encode(p, x, u, remaining) for x in ligs])
            a, rec0 = self.ppo.agents[0].choose(s, lf, mask)
            selected = ligs[a]
            if self.args.setting == "cooperative":
                if len(pockets) == 1:
                    # A missing edit proposal remains explicit; no fabricated second-agent update.
                    b = 0
                    rec1 = None
                else:
                    pf = np.array([encode(x, selected, u, remaining) for x in pockets])
                    b, rec1 = self.ppo.agents[1].choose(
                        encode(p, selected, u, remaining),
                        pf,
                        np.ones(len(pockets), bool),
                    )
                pairp = pockets[b]
                result = self.oracle.evaluate(pairp, selected, target, self.args.seed)
                old = u
                if result["feasible"]:
                    p, l, u = pairp, selected, float(result["utility"])
                    self.training_successes += int(a != 0 or b != 0)
                r = (
                    u - old - self.cfg["reward"]["action_cost"]
                    if result["feasible"]
                    else self.cfg["reward"]["infeasible_reward"]
                )
                r0 = r1 = r
            else:
                pf = np.array([encode(x, selected, u, remaining) for x in states])
                b, rec1 = self.ppo.agents[1].choose(
                    encode(p, selected, u, remaining), pf, np.ones(len(states), bool)
                )
                pairp = states[b]
                result = self.oracle.evaluate(pairp, selected, target, self.args.seed)
                r0 = float(result["utility"]) if result["feasible"] else -20.0
                r1 = -r0
                if result["feasible"]:
                    l = selected
                    u = r0
                    self.training_successes += int(a != 0)
            records[0].append(rec0)
            records[1].append(rec1)
            rewards[0].append(r0)
            rewards[1].append(r1)
            trace.append(
                {
                    "step": k,
                    "ligand_action": a,
                    "partner_action": b,
                    "protein": pairp,
                    "ligand": selected,
                    "feasible": result["feasible"],
                    "utility": result["utility"],
                    "rewards": [r0, r1],
                }
            )
            self.step += 1
        self.ppo.add_episode(records, rewards)
        append_json(
            self.out / "training.jsonl",
            {
                "episode": self.episode,
                "target": target["id"],
                "steps": trace,
                "return_ligand": sum(rewards[0]),
                "return_partner": sum(rewards[1]),
            },
        )

    def cooperative_eval(self, target, method, seed):
        p, l = target["protein"], target["ligand"]
        u, initial = self.initial(target)
        best = (u, p, l)
        queries = 0
        path = []
        budget = self.cfg["evaluation"]["score_budget_per_target"]
        h = self.cfg["evaluation"]["horizon"]
        for k in range(h):
            ligs, pockets = self.proposals.generate(
                p, l, target, seed + k, cooperative=True
            )
            mask = chemical_mask(ligs, target["ligand"], self.cfg["constraints"])
            mask[0] = True
            pairs = [
                (i, j)
                for i in range(len(ligs))
                for j in range(len(pockets))
                if mask[i] and (i or j)
            ]
            if method == "ligand_only":
                pairs = [x for x in pairs if x[1] == 0]
            elif method == "greedy_alternating":
                pairs = (
                    [x for x in pairs if x[1] == 0]
                    if k % 2 == 0
                    else [x for x in pairs if x[0] == 0]
                )
            elif method == "learned":
                s = torch.tensor(encode(p, l, u, (h - k) / h))
                lf = torch.tensor(
                    np.array([encode(p, x, u, (h - k) / h) for x in ligs])
                )
                with torch.no_grad():
                    lp = (
                        self.ppo.agents[0]
                        .distribution(s, lf, torch.tensor(mask))
                        .logits
                    )
                    scores = {}
                    for i in range(len(ligs)):
                        pf = torch.tensor(
                            np.array(
                                [encode(x, ligs[i], u, (h - k) / h) for x in pockets]
                            )
                        )
                        pp = (
                            self.ppo.agents[1]
                            .distribution(
                                torch.tensor(encode(p, ligs[i], u, (h - k) / h)),
                                pf,
                                torch.ones(len(pockets), dtype=torch.bool),
                            )
                            .logits
                        )
                        for j in range(len(pockets)):
                            scores[i, j] = float(lp[i] + pp[j])
                pairs.sort(key=lambda x: scores[x], reverse=True)
            if method != "learned":
                np.random.default_rng(seed + k).shuffle(pairs)
            for i, j in pairs[: min(8, budget - queries)]:
                row = self.oracle.evaluate(pockets[j], ligs[i], target, seed)
                queries += 1
                path.append(
                    {
                        "protein": pockets[j],
                        "ligand": ligs[i],
                        "feasible": row["feasible"],
                        "utility": row["utility"],
                    }
                )
                if row["feasible"] and row["utility"] > best[0]:
                    best = (float(row["utility"]), pockets[j], ligs[i])
            u, p, l = best
            if queries >= budget:
                break
        return {
            "method": method,
            "target": target["id"],
            "initial_utility": initial["utility"],
            "reference_feasible": initial["feasible"],
            "best_utility": best[0],
            "protein": best[1],
            "ligand": best[2],
            "optimization_queries": queries,
            "observations": path,
        }

    def competitive_eval(self, target, method, seed):
        states, heldout = self.ensemble(target)
        h = self.cfg["evaluation"]["horizon"]
        pool = [target["ligand"]]
        # Equal proposal budget: all methods generate h * ligands fresh candidates.
        for k in range(h):
            context = (
                target["protein"] if method == "rigid" else states[k % len(states)]
            )
            ligs, _ = self.proposals.generate(
                context, target["ligand"], target, seed + k, cooperative=False
            )
            pool.extend(ligs[1:])
        mask = chemical_mask(pool, target["ligand"], self.cfg["constraints"])
        indices = [i for i in range(1, len(pool)) if mask[i]]
        if method == "learned":
            s = torch.tensor(encode(target["protein"], target["ligand"], 0, 1))
            a = torch.tensor(
                np.array([encode(target["protein"], x, 0, 1) for x in pool])
            )
            with torch.no_grad():
                logits = (
                    self.ppo.agents[0]
                    .distribution(s, a, torch.tensor(mask))
                    .logits.numpy()
                )
            indices.sort(key=lambda i: float(logits[i]), reverse=True)
        elif method == "greedy_geometry":
            indices.sort(
                key=lambda i: float(encode(target["protein"], pool[i])[29]),
                reverse=False,
            )
        else:
            np.random.default_rng(seed).shuffle(indices)
        n = self.cfg["evaluation"]["score_budget_per_target"] // len(states)
        indices = indices[:n]
        rows = []
        for i in indices:
            scores = [self.oracle.evaluate(p, pool[i], target, seed) for p in states]
            vals = [float(x["utility"]) if x["feasible"] else -20.0 for x in scores]
            rows.append(
                {
                    "ligand": pool[i],
                    "protein": target["protein"],
                    "values": vals,
                    "worst": min(vals),
                    "mean": float(np.mean(vals)),
                    "all_feasible": all(x["feasible"] for x in scores),
                }
            )
        if not rows:
            raise RuntimeError("No eligible generated candidates in evaluation")
        # Every method shares the same exact training-ensemble screening objective.
        selected = max(rows, key=lambda r: r["worst"])
        held = [
            self.oracle.evaluate(p, selected["ligand"], target, seed) for p in heldout
        ]
        vals = [float(x["utility"]) if x["feasible"] else -20.0 for x in held]
        return {
            "method": method,
            "target": target["id"],
            "best_utility": selected["worst"],
            "protein": target["protein"],
            "ligand": selected["ligand"],
            "training_state_utilities": selected["values"],
            "heldout_mean": float(np.mean(vals)),
            "heldout_worst": min(vals),
            "heldout_feasibility": sum(x["feasible"] for x in held) / len(held),
            "optimization_queries": len(rows) * len(states),
            "heldout_queries": len(held),
            "observations": rows,
        }

    def evaluate_split(self, split, methods, label):
        results = []
        folder = self.out / "evaluation" / label
        folder.mkdir(parents=True, exist_ok=True)
        for ti, target in enumerate(self.manifest["splits"][split]):
            for method in methods:
                file = folder / f'{target["id"]}-{method}.json'
                if file.exists():
                    result = json.loads(file.read_text())
                else:
                    start = time.monotonic()
                    before = (
                        self.oracle.calls,
                        self.proposals.generated,
                        self.proposals.pocket_generated,
                    )
                    try:
                        result = (
                            self.cooperative_eval
                            if self.args.setting == "cooperative"
                            else self.competitive_eval
                        )(target, method, self.args.seed + 10000000 + ti * 100)
                        if split == "test":
                            result["final_docking"] = self.oracle.evaluate(
                                result["protein"],
                                result["ligand"],
                                target,
                                self.args.seed,
                                dock=True,
                            )
                        result.update(ok=True, seconds=time.monotonic() - start)
                    except Exception as e:
                        result = {
                            "ok": False,
                            "target": target["id"],
                            "method": method,
                            "error": repr(e),
                            "best_utility": -20.0,
                        }
                    result["costs"] = {
                        "oracle_requests": self.oracle.calls - before[0],
                        "generated_ligands": self.proposals.generated - before[1],
                        "pocket_proposals": self.proposals.pocket_generated - before[2],
                    }
                    atomic_json(file, result)
                results.append(result)
                self.save_status(
                    stage="evaluation",
                    evaluation_label=label,
                    evaluation_completed=len(results),
                    evaluation_expected=len(self.manifest["splits"][split])
                    * len(methods),
                )
        atomic_json(folder / "results.json", results)
        return results

    def run(self):
        try:
            self.save_status(stage="training")
            while self.episode < self.cfg["training"]["episodes"]:
                target = self.manifest["splits"]["train"][
                    int(self.rng.integers(len(self.manifest["splits"]["train"])))
                ]
                self.episode_train(target)
                self.episode += 1
                if self.episode % self.cfg["training"]["update_every"] == 0:
                    stats = self.ppo.update()
                    append_json(
                        self.out / "updates.jsonl", dict(episode=self.episode, **stats)
                    )
                self.checkpoint()
                self.save_status(stage="training")
                if self.episode >= 16 and self.training_successes == 0:
                    raise RuntimeError(
                        "No feasible nontrivial training transitions after 16 episodes; inspect validity, do not claim RL training succeeded"
                    )
                if self.episode % self.cfg["training"]["validate_every"] == 0:
                    self.proposals.close()  # release GPU contexts before MD/inference for validation
                    rows = self.evaluate_split(
                        "val", ["learned"], f"validation-{self.episode:04d}"
                    )
                    value = float(np.mean([r["best_utility"] for r in rows]))
                    self.last_validation = value
                    if value > self.best_validation:
                        self.best_validation = value
                        torch.save(
                            self.ppo.agents.state_dict(), self.out / "best_policy.pt"
                        )
                    self.checkpoint()
            self.ppo.update()
            self.checkpoint()
            if not (self.out / "best_policy.pt").exists():
                raise RuntimeError("Validation never selected a checkpoint")
            self.ppo.agents.load_state_dict(
                torch.load(
                    self.out / "best_policy.pt", map_location="cpu", weights_only=True
                )
            )
            # Test manifest is only consulted after a validation-selected policy has been frozen.
            methods = (
                ["learned", "ligand_only", "greedy_alternating", "random_joint"]
                if self.args.setting == "cooperative"
                else ["learned", "rigid", "uniform_ensemble", "greedy_geometry"]
            )
            self.proposals.close()
            rows = self.evaluate_split("test", methods, "test")
            valid = sum(r.get("ok", False) for r in rows)
            if valid == 0:
                raise RuntimeError(
                    "Every held-out evaluation failed; no valid benchmark result"
                )
            self.save_status(
                stage="complete",
                best_validation=self.best_validation,
                evaluation_successes=valid,
                evaluation_failures=len(rows) - valid,
            )
        except Exception as e:
            self.save_status(stage="failed", error=repr(e))
            raise
        finally:
            self.proposals.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["cooperative", "competitive"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    Campaign(a).run()


if __name__ == "__main__":
    main()
