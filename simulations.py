import random
from collections import defaultdict
from utils import compute_payoff
from players import Learner, Learner_Extended_History, Cooperator, Defector, TitForTat, Unforgiving, Pheromones


STRATEGY_LABEL = {
    Cooperator:                "Cooperator",
    Defector:                  "Defector",
    TitForTat:                 "TitForTat",
    Unforgiving:               "Unforgiving",
    Pheromones:                "Pheromones",
    Learner:                   "Learner",
    Learner_Extended_History:  "LearnerExt",
}


class Simulation:

    def __init__(self, agents, ticks_per_episode=200, episodes=20):
        self.agents = agents
        self.ticks_per_episode = ticks_per_episode
        self.episodes = episodes
        self.current_tick = 0
        self.current_episode = 0

        # per-tick
        self.tick_coop_ratio = []

        # per-episode  {strategy_label: [ep0_value, ep1_value, ...]}
        self.ep_avg_score  = defaultdict(list)
        self.ep_coop_ratio = defaultdict(list)
        self.ep_epsilon    = []

    # ---------------------------------------------------------------
    def partner_up(self):
        shuffled = self.agents[:]
        random.shuffle(shuffled)
        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        return pairs

    # ---------------------------------------------------------------
    def run_episode(self):
        strat_coop  = defaultdict(int)
        strat_total = defaultdict(int)

        for _ in range(self.ticks_per_episode):
            pairs = self.partner_up()
            tick_coop = 0
            tick_actions = 0

            for a1, a2 in pairs:
                a1.play(a2); a2.play(a1)

                for agent in (a1, a2):
                    lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
                    strat_total[lbl] += 1
                    if agent.last_action == 0:
                        strat_coop[lbl] += 1

                tick_coop    += (a1.last_action == 0) + (a2.last_action == 0)
                tick_actions += 2

                p1 = compute_payoff(a1.last_action, a2.last_action)
                p2 = compute_payoff(a2.last_action, a1.last_action)
                a1.score += p1; a2.score += p2

                a1.observe(a2.last_action); a2.observe(a1.last_action)

                if isinstance(a1, (Learner, Learner_Extended_History)): a1.learn(p1)
                if isinstance(a2, (Learner, Learner_Extended_History)): a2.learn(p2)

            self.tick_coop_ratio.append(
                tick_coop / tick_actions if tick_actions > 0 else 0)
            self.current_tick += 1
            Pheromones.evaporate()  # decay cooperation signal each tick

        # ---- per-episode aggregates ----
        strat_scores = defaultdict(list)
        for agent in self.agents:
            lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
            strat_scores[lbl].append(agent.score)

        for lbl, sc in strat_scores.items():
            self.ep_avg_score[lbl].append(sum(sc) / len(sc))

        for lbl in strat_total:
            r = strat_coop[lbl] / strat_total[lbl] if strat_total[lbl] else 0
            self.ep_coop_ratio[lbl].append(r)

        for agent in self.agents:
            if isinstance(agent, (Learner, Learner_Extended_History)):
                self.ep_epsilon.append(agent.epsilon); break
        else:
            self.ep_epsilon.append(None)

    # ---------------------------------------------------------------
    def reset_episode(self):
        for agent in self.agents:
            agent.score = 0
            agent.partner = None
            agent.last_action = None
            if isinstance(agent, (Learner, Learner_Extended_History)):
                agent.prev_pid    = None
                agent.prev_state  = None
                agent.prev_action = None
        self.current_tick = 0

    # ---------------------------------------------------------------
    def run(self, print_every=100):
        for ep in range(self.episodes):
            self.current_episode = ep + 1
            self.run_episode()

            if (ep + 1) % print_every == 0:
                self._print_episode_stats()

            if ep < self.episodes - 1:
                for agent in self.agents:
                    if isinstance(agent, (Learner, Learner_Extended_History)):
                        agent.decay_epsilon()
                self.reset_episode()

        self._print_final_summary()

    # ---------------------------------------------------------------
    def _print_episode_stats(self):
        print(f"\n===== EPISODE {self.current_episode} =====")
        for lbl in self.ep_avg_score:
            sc = self.ep_avg_score[lbl][-1]
            cr = self.ep_coop_ratio.get(lbl, [0])[-1]
            print(f"  {lbl:12s}  avg_score={sc:7.1f}  coop={cr:.2f}")
        eps = self.ep_epsilon[-1] if self.ep_epsilon else None
        if eps is not None:
            print(f"  epsilon = {eps:.4f}")

        # Per-partner policy summary for each Learner
        self._print_learner_policies()

    def _print_learner_policies(self):
        """
        For each Learner, show the preferred action vs each specific partner,
        grouped by partner strategy type.
        """
        # build a pid -> strategy_label map
        pid_to_label = {a.id: STRATEGY_LABEL.get(type(a), type(a).__name__)
                        for a in self.agents}

        sym = {-1: "?", 0: "C", 1: "D"}

        for agent in self.agents:
            if not isinstance(agent, (Learner, Learner_Extended_History)):
                continue
            print(f"\n  Learner {agent.id} per-partner policy:")
            # group partners by strategy
            by_type = defaultdict(list)
            for pid, qt in agent.qtables.items():
                lbl = pid_to_label.get(pid, f"id={pid}")
                by_type[lbl].append((pid, qt))

            for lbl, entries in sorted(by_type.items()):
                for pid, qt in entries:
                    lines = []
                    for state in sorted(qt):
                        q = qt[state]
                        pref = "C" if q[0] >= q[1] else "D"
                        lines.append(
                            f"(I={sym[state[0]]},opp={sym[state[1]]})→{pref}"
                            f"[{q[0]:.1f},{q[1]:.1f}]")
                    print(f"    vs {lbl:12s} (id={pid}): "
                          + "  ".join(lines))

    def _print_final_summary(self):
        print("\n===== FINAL SUMMARY =====")
        w = min(100, self.episodes)
        for lbl in self.ep_avg_score:
            sc = self.ep_avg_score[lbl]
            cr = self.ep_coop_ratio.get(lbl, [])
            print(f"  {lbl:12s}  "
                  f"avg_score (last {w} ep): {sum(sc[-w:])/w:7.1f}  "
                  f"coop (last {w} ep): {sum(cr[-w:])/w:.3f}")
        self._print_learner_policies()

    # ---------------------------------------------------------------
    def plot(self, smooth_window=50, save_path=None):
        """
        Save each chart as a separate image inside a subfolder.
        If save_path="results.png", plots are saved in results/ as:
            1_global_coop.png, 2_avg_score.png, 3_coop_ratio.png,
            4_epsilon.png, 5_learner_policy.png, 6_learner_return.png
        If save_path is None, each figure is shown interactively.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import os
        except ImportError:
            print("matplotlib not installed — pip install matplotlib")
            return

        def smooth(data, w):
            if w <= 1 or len(data) < w:
                return list(range(len(data))), data
            kernel = np.ones(w) / w
            sm = np.convolve(data, kernel, mode="valid").tolist()
            return list(range(w - 1, len(data))), sm

        # ── Output folder ────────────────────────────────────────────
        if save_path:
            base, _ = os.path.splitext(save_path)
            folder = base
            os.makedirs(folder, exist_ok=True)
        else:
            folder = None

        def save_or_show(fig, suffix):
            if folder:
                path = os.path.join(folder, f"{suffix}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"Saved → {path}")
            else:
                fig.show()
            plt.close(fig)

        eps_x = list(range(1, self.episodes + 1))

        # Fixed colours — consistent across all simulations
        FIXED_COLORS = {
            "Cooperator":  "#1f77b4",   # blue
            "Defector":    "#ff7f0e",   # orange
            "TitForTat":   "#2ca02c",   # green
            "Unforgiving": "#8c564b",   # brown
            "Pheromones":  "#e377c2",   # pink
            "Learner":     "#d62728",   # red
            "LearnerExt":  "#9467bd",   # purple
        }
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        lbl_color = {lbl: FIXED_COLORS.get(lbl, prop_cycle[i % len(prop_cycle)])
                     for i, lbl in enumerate(self.ep_avg_score)}

        # ── 1. Global coop ratio per tick ────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        raw = self.tick_coop_ratio
        x_sm, sm = smooth(raw, smooth_window)
        ax.plot(list(range(len(raw))), raw, alpha=0.12,
                color="steelblue", linewidth=0.6)
        ax.plot(x_sm, sm, color="steelblue", linewidth=1.8,
                label=f"smoothed (w={smooth_window})")
        ax.set_xlabel("Tick"); ax.set_ylabel("Cooperation ratio")
        ax.set_title("Global cooperation ratio (per tick)")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
        fig.tight_layout()
        save_or_show(fig, "1_global_coop")

        # ── 2. Avg score per strategy per episode ────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        for lbl, scores in self.ep_avg_score.items():
            ax.plot(eps_x, scores, label=lbl,
                    color=lbl_color[lbl], linewidth=1.4)
        ax.set_xlabel("Episode"); ax.set_ylabel("Avg score")
        ax.set_title("Average score per strategy")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_or_show(fig, "2_avg_score")

        # ── 3. Coop ratio per strategy per episode ───────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        w2 = max(1, smooth_window // 5)
        for lbl, crs in self.ep_coop_ratio.items():
            x_s, sm_crs = smooth(crs, w2)
            ax.plot(x_s, sm_crs, label=lbl,
                    color=lbl_color[lbl], linewidth=1.4)
        ax.set_xlabel("Episode"); ax.set_ylabel("Cooperation ratio")
        ax.set_title(f"Cooperation ratio per strategy (smoothed w={w2})")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
        fig.tight_layout()
        save_or_show(fig, "3_coop_ratio")

        # ── 4. Epsilon ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        eps_vals = [e for e in self.ep_epsilon if e is not None]
        if eps_vals:
            ax.plot(range(1, len(eps_vals) + 1), eps_vals,
                    color="darkorange", linewidth=1.5)
            ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
            ax.set_title("Learner exploration rate (ε)")
            ax.set_ylim(-0.02, max(eps_vals) * 1.05)
        else:
            ax.text(0.5, 0.5, "No Learner in simulation",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Epsilon (N/A)")
        fig.tight_layout()
        save_or_show(fig, "4_epsilon")

        # ── 5. Learner final policy by opponent type ──────────────────
        # Separate subplots for Learner and LearnerExt
        pid_to_label = {a.id: STRATEGY_LABEL.get(type(a), type(a).__name__)
                        for a in self.agents}

        pref_learner     = defaultdict(lambda: [0, 0])
        pref_learner_ext = defaultdict(lambda: [0, 0])

        for agent in self.agents:
            if isinstance(agent, Learner_Extended_History):
                target = pref_learner_ext
            elif isinstance(agent, Learner):
                target = pref_learner
            else:
                continue
            for pid, qt in agent.qtables.items():
                lbl = pid_to_label.get(pid, "?")
                c_score = sum(max(0, v[0] - v[1]) for v in qt.values())
                d_score = sum(max(0, v[1] - v[0]) for v in qt.values())
                if c_score >= d_score:
                    target[lbl][0] += 1
                else:
                    target[lbl][1] += 1

        has_learner     = bool(pref_learner)
        has_learner_ext = bool(pref_learner_ext)
        n_panels = (1 if has_learner else 0) + (1 if has_learner_ext else 0)

        if n_panels == 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No Learner data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Learner policy by opponent type (N/A)")
            fig.tight_layout()
            save_or_show(fig, "5_learner_policy")
        else:
            fig, axes = plt.subplots(1, n_panels,
                                     figsize=(7 * n_panels, 4),
                                     squeeze=False)
            panel_idx = 0
            width = 0.35

            def _draw_policy_panel(ax_, pref_dict, title):
                types = sorted(pref_dict)
                x_pos = range(len(types))
                ax_.bar([x - width/2 for x in x_pos],
                        [pref_dict[t][0] for t in types], width,
                        label="Prefer C", color="steelblue")
                ax_.bar([x + width/2 for x in x_pos],
                        [pref_dict[t][1] for t in types], width,
                        label="Prefer D", color="tomato")
                ax_.set_xticks(list(x_pos))
                ax_.set_xticklabels(types, fontsize=9)
                ax_.set_ylabel("Number of agent↔partner pairs")
                ax_.set_title(title)
                ax_.legend(fontsize=8)

            if has_learner:
                _draw_policy_panel(axes[0][panel_idx], pref_learner,
                                   "Learner final policy by opponent type")
                panel_idx += 1
            if has_learner_ext:
                _draw_policy_panel(axes[0][panel_idx], pref_learner_ext,
                                   "LearnerExt final policy by opponent type")

            fig.tight_layout()
            save_or_show(fig, "5_learner_policy")

        # ── 6. Learner expected return by opponent type ───────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ev_by_type = defaultdict(list)
        for agent in self.agents:
            if not isinstance(agent, (Learner, Learner_Extended_History)):
                continue
            for pid, qt in agent.qtables.items():
                lbl = pid_to_label.get(pid, "?")
                if qt:
                    ev = max(max(v) for v in qt.values())
                    ev_by_type[lbl].append(ev)

        if ev_by_type:
            types  = sorted(ev_by_type)
            means  = [sum(ev_by_type[t]) / len(ev_by_type[t]) for t in types]
            colors_bar = [lbl_color.get(t, "gray") for t in types]
            ax.bar(types, means, color=colors_bar)
            ax.set_ylabel("Max Q-value (proxy for expected return)")
            ax.set_title("Learner expected return by opponent type")
        else:
            ax.text(0.5, 0.5, "No Learner data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Learner expected return (N/A)")
        fig.tight_layout()
        save_or_show(fig, "6_learner_return")