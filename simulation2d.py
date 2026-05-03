import random
import numpy as np
from collections import defaultdict
from utils import compute_payoff
from players import (Learner, Learner_Extended_History,
                     Cooperator, Defector, TitForTat,
                     Unforgiving, Pheromones)

STRATEGY_LABEL = {
    Cooperator: "Cooperator",
    Defector: "Defector",
    TitForTat: "TitForTat",
    Unforgiving: "Unforgiving",
    Pheromones: "Pheromones",
    Learner: "Learner",
    Learner_Extended_History: "LearnerExt",
}

FIXED_COLORS = {
    "Cooperator": "#1f77b4",
    "Defector": "#ff7f0e",
    "TitForTat": "#2ca02c",
    "Unforgiving": "#8c564b",
    "Pheromones": "#e377c2",
    "Learner": "#d62728",
    "LearnerExt": "#9467bd",
}


class Grid:
    """
    Toroidal 2D grid (rows x cols).
    Each cell holds at most one agent (None if empty).
    Pheromone field is a float array of the same shape.
    """

    def __init__(self, rows, cols, diffusion=0.3, evaporation=0.9):
        self.rows = rows
        self.cols = cols
        self.diffusion = diffusion  # fraction of pheromone that spreads to neighbours
        self.evaporation = evaporation  # fraction retained each tick
        self.cells = [[None] * cols for _ in range(rows)]
        self.pheromone = np.zeros((rows, cols), dtype=float)

    # ── cell access ────────────────────────────────────────────────
    def _wrap(self, r, c):
        return r % self.rows, c % self.cols

    def get(self, r, c):
        r, c = self._wrap(r, c)
        return self.cells[r][c]

    def place(self, agent, r, c):
        r, c = self._wrap(r, c)
        self.cells[r][c] = agent
        agent.row, agent.col = r, c

    def move(self, agent, new_r, new_c):
        new_r, new_c = self._wrap(new_r, new_c)
        self.cells[agent.row][agent.col] = None
        self.cells[new_r][new_c] = agent
        agent.row, agent.col = new_r, new_c

    def empty_cells(self):
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols) if self.cells[r][c] is None]

    def neighbours(self, r, c):
        """Von Neumann neighbourhood (4 adjacent cells)."""
        return [self._wrap(r + dr, c + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

    def neighbour_agents(self, r, c):
        return [self.get(nr, nc)
                for nr, nc in self.neighbours(r, c)
                if self.get(nr, nc) is not None]

    # ── pheromone ──────────────────────────────────────────────────
    def deposit(self, r, c, amount):
        self.pheromone[r, c] += amount

    def diffuse_and_evaporate(self):
        """
        Each cell spreads `diffusion` fraction of its pheromone equally
        to its 4 neighbours, then the whole field evaporates.
        """
        kernel = np.array([[0, self.diffusion / 4, 0],
                           [self.diffusion / 4, 0, self.diffusion / 4],
                           [0, self.diffusion / 4, 0]])
        from scipy.ndimage import convolve
        spread = convolve(self.pheromone, kernel, mode='wrap')
        self.pheromone = (self.pheromone * (1 - self.diffusion) + spread) * self.evaporation
        self.pheromone = np.clip(self.pheromone, 0, None)

    def pheromone_at(self, r, c):
        return float(self.pheromone[r, c])

    def best_pheromone_neighbour(self, r, c):
        """Return (nr, nc) of the neighbour cell with highest pheromone."""
        nbrs = self.neighbours(r, c)
        return max(nbrs, key=lambda rc: self.pheromone[rc[0], rc[1]])


# ── Pygame colour palette (matches FIXED_COLORS) ──────────────────────────────
PYGAME_COLORS = {
    "Cooperator": (31, 119, 180),
    "Defector": (255, 127, 14),
    "TitForTat": (44, 160, 44),
    "Unforgiving": (140, 86, 75),
    "Pheromones": (227, 119, 194),
    "Learner": (214, 39, 40),
    "LearnerExt": (148, 103, 189),
}
PYGAME_BG = (15, 15, 20)  # dark background
PYGAME_GRID = (30, 30, 38)  # faint grid lines
PYGAME_TEXT = (220, 220, 220)
PHERO_HOT = (255, 200, 50)  # colour tint for high pheromone


class Simulation2D:
    """
    Spatial simulation on a toroidal grid.

    Movement rules (per tick, before playing):
      - Pheromones agents: move toward the adjacent cell with highest
        pheromone if above sniff_threshold, else random wiggle.
      - All other agents: random wiggle (move to a random empty neighbour).

    Pairing: each agent plays against one randomly chosen neighbour.
    Agents without neighbours skip the tick.

    Pheromone emission: Pheromones agents deposit `phero_drop` on their
    cell each time they cooperate.

    chemotaxis : bool (default True)
      If True, Pheromones agents follow the pheromone gradient when
      phi >= sniff_threshold. If False, they move randomly like all
      other agents (ablation experiment).

    Live rendering (pygame):
      Pass live=True to run() to open a pygame window.
      Each cell is drawn as a coloured square (strategy colour).
      The pheromone field is overlaid as a warm glow.
      A legend + episode/tick counter is shown in the sidebar.
      Press Q or close the window to abort early.
      live_fps   – max frames per second rendered (default 30).
      live_skip  – render every N ticks (default 1; increase for speed).
    """

    def __init__(self, agents,
                 grid_rows=30, grid_cols=30,
                 ticks_per_episode=200, episodes=20,
                 diffusion=0.3, evaporation=0.9,
                 phero_drop=3.0, sniff_threshold=1.0,
                 chemotaxis=True):

        self.agents = agents
        self.ticks_per_episode = ticks_per_episode
        self.episodes = episodes
        self.sniff_threshold = sniff_threshold
        self.phero_drop = phero_drop
        self.chemotaxis = chemotaxis

        if len(agents) > grid_rows * grid_cols:
            raise ValueError(
                f"Too many agents ({len(agents)}) for grid "
                f"{grid_rows}x{grid_cols}={grid_rows * grid_cols} cells.")

        self.grid = Grid(grid_rows, grid_cols,
                         diffusion=diffusion,
                         evaporation=evaporation)
        self._place_agents_randomly()

        self.current_tick = 0
        self.current_episode = 0

        self.tick_coop_ratio = []
        self.ep_avg_score = defaultdict(list)
        self.ep_coop_ratio = defaultdict(list)
        self.ep_epsilon = []

    # ── initialisation ─────────────────────────────────────────────
    def _place_agents_randomly(self):
        cells = self.grid.empty_cells()
        random.shuffle(cells)
        for agent, (r, c) in zip(self.agents, cells):
            self.grid.place(agent, r, c)

    # ── movement ───────────────────────────────────────────────────
    def _move_agent(self, agent):
        r, c = agent.row, agent.col
        nbrs = self.grid.neighbours(r, c)
        empty = [(nr, nc) for nr, nc in nbrs
                 if self.grid.get(nr, nc) is None]

        if self.chemotaxis and isinstance(agent, Pheromones):
            phero = self.grid.pheromone_at(r, c)
            if phero >= self.sniff_threshold and empty:
                # move toward highest-pheromone empty neighbour
                target = max(empty,
                             key=lambda rc: self.grid.pheromone[rc[0], rc[1]])
                self.grid.move(agent, *target)
                return
        # random wiggle
        if empty:
            self.grid.move(agent, *random.choice(empty))

    # ── tick ───────────────────────────────────────────────────────
    def _run_tick(self):
        # 1. Move all agents
        agent_order = self.agents[:]
        random.shuffle(agent_order)
        for agent in agent_order:
            self._move_agent(agent)

        # 2. Pair each agent with one random neighbour (each pair plays once)
        played = set()
        pairs = []
        for agent in agent_order:
            if id(agent) in played:
                continue
            nbrs = self.grid.neighbour_agents(agent.row, agent.col)
            candidates = [n for n in nbrs if id(n) not in played]
            if not candidates:
                continue
            partner = random.choice(candidates)
            pairs.append((agent, partner))
            played.add(id(agent))
            played.add(id(partner))

        # 3. Play all pairs
        tick_coop = 0
        tick_actions = 0
        strat_coop = defaultdict(int)
        strat_total = defaultdict(int)

        for a1, a2 in pairs:
            a1.play(a2);
            a2.play(a1)

            for agent in (a1, a2):
                lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
                strat_total[lbl] += 1
                if agent.last_action == 0:
                    strat_coop[lbl] += 1

            tick_coop += (a1.last_action == 0) + (a2.last_action == 0)
            tick_actions += 2

            p1 = compute_payoff(a1.last_action, a2.last_action)
            p2 = compute_payoff(a2.last_action, a1.last_action)
            a1.score += p1;
            a2.score += p2

            a1.observe(a2.last_action);
            a2.observe(a1.last_action)

            # Pheromone emission: deposit if cooperated
            for agent, payoff in ((a1, p1), (a2, p2)):
                if isinstance(agent, Pheromones) and agent.last_action == 0:
                    self.grid.deposit(agent.row, agent.col, self.phero_drop)

            if isinstance(a1, (Learner, Learner_Extended_History)): a1.learn(p1)
            if isinstance(a2, (Learner, Learner_Extended_History)): a2.learn(p2)

        # 4. Diffuse and evaporate pheromone field
        self.grid.diffuse_and_evaporate()

        self.tick_coop_ratio.append(
            tick_coop / tick_actions if tick_actions > 0 else 0)
        self.current_tick += 1

        return strat_coop, strat_total

    # ── episode ────────────────────────────────────────────────────
    def run_episode(self):
        ep_strat_coop = defaultdict(int)
        ep_strat_total = defaultdict(int)

        for _ in range(self.ticks_per_episode):
            sc, st = self._run_tick()
            for k in sc: ep_strat_coop[k] += sc[k]
            for k in st: ep_strat_total[k] += st[k]

        # per-episode aggregates
        strat_scores = defaultdict(list)
        for agent in self.agents:
            lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
            strat_scores[lbl].append(agent.score)

        for lbl, sc in strat_scores.items():
            self.ep_avg_score[lbl].append(sum(sc) / len(sc))

        for lbl in ep_strat_total:
            r = (ep_strat_coop[lbl] / ep_strat_total[lbl]
                 if ep_strat_total[lbl] else 0)
            self.ep_coop_ratio[lbl].append(r)

        for agent in self.agents:
            if isinstance(agent, (Learner, Learner_Extended_History)):
                self.ep_epsilon.append(agent.epsilon);
                break
        else:
            self.ep_epsilon.append(None)

    # ── reset ──────────────────────────────────────────────────────
    def reset_episode(self):
        for agent in self.agents:
            agent.score = 0
            agent.partner = None
            agent.last_action = None
            if isinstance(agent, (Learner, Learner_Extended_History)):
                agent.prev_pid = None
                agent.prev_state = None
                agent.prev_action = None
            if isinstance(agent, Unforgiving):
                agent.betrayed_by = set()
        self.current_tick = 0

    # ── run ────────────────────────────────────────────────────────
    def run(self, print_every=100, live=False, live_fps=30, live_skip=1):
        """
        Run the simulation.

        Parameters
        ----------
        print_every : int
            Print stats every N episodes (console).
        live        : bool
            Open a pygame window for real-time visualisation.
        live_fps    : int
            Maximum frames per second rendered.
        live_skip   : int
            Render every N ticks (1 = every tick; higher = faster sim).
        """
        if live:
            self._run_live(print_every=print_every,
                           fps=live_fps, skip=live_skip)
        else:
            self._run_headless(print_every=print_every)

    def _run_headless(self, print_every=100):
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

    # ── pygame live rendering ──────────────────────────────────────
    def _run_live(self, print_every=100, fps=30, skip=1):
        try:
            import pygame
        except ImportError:
            print("pygame not installed — pip install pygame")
            print("Falling back to headless mode.")
            self._run_headless(print_every=print_every)
            return

        pygame.init()
        pygame.display.set_caption("Prisoner's Dilemma — 2D Simulation")

        SIDEBAR = 220
        CELL = max(4, min(24, 720 // max(self.grid.rows, self.grid.cols)))
        GRID_W = self.grid.cols * CELL
        GRID_H = self.grid.rows * CELL
        WIN_W = GRID_W + SIDEBAR
        WIN_H = max(GRID_H, 400)

        screen = pygame.display.set_mode((WIN_W, WIN_H))
        clock = pygame.time.Clock()
        font_s = pygame.font.SysFont("monospace", 11)
        font_m = pygame.font.SysFont("monospace", 13, bold=True)

        # pre-build strategy→colour lookup
        agent_color = {}
        for agent in self.agents:
            lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
            agent_color[id(agent)] = PYGAME_COLORS.get(lbl, (180, 180, 180))

        # pheromone surface (reused each frame)
        phero_surf = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)

        def draw_frame(ep, tick):
            screen.fill(PYGAME_BG)

            # --- pheromone glow ---
            phero_surf.fill((0, 0, 0, 0))
            pmax = self.grid.pheromone.max()
            if pmax > 0:
                for r in range(self.grid.rows):
                    for c in range(self.grid.cols):
                        v = self.grid.pheromone[r, c] / pmax
                        if v > 0.01:
                            alpha = int(min(200, v * 200))
                            pr = int(PHERO_HOT[0] * v)
                            pg = int(PHERO_HOT[1] * v)
                            pb = int(PHERO_HOT[2] * v)
                            pygame.draw.rect(
                                phero_surf, (pr, pg, pb, alpha),
                                (c * CELL, r * CELL, CELL, CELL))
            screen.blit(phero_surf, (0, 0))

            # --- agents ---
            for agent in self.agents:
                color = agent_color[id(agent)]
                r, c = agent.row, agent.col
                rect = pygame.Rect(c * CELL + 1, r * CELL + 1,
                                   CELL - 2, CELL - 2)
                pygame.draw.rect(screen, color, rect, border_radius=max(1, CELL // 4))

            # --- faint grid lines (only if cells are big enough) ---
            if CELL >= 8:
                for r in range(self.grid.rows + 1):
                    pygame.draw.line(screen, PYGAME_GRID,
                                     (0, r * CELL), (GRID_W, r * CELL))
                for c in range(self.grid.cols + 1):
                    pygame.draw.line(screen, PYGAME_GRID,
                                     (c * CELL, 0), (c * CELL, GRID_H))

            # --- sidebar ---
            sx = GRID_W + 10
            sy = 10
            title = font_m.render("2D Sim — Live", True, PYGAME_TEXT)
            screen.blit(title, (sx, sy));
            sy += 22

            for line in (
                    f"Episode : {ep}/{self.episodes}",
                    f"Tick    : {tick}/{self.ticks_per_episode}",
                    f"Agents  : {len(self.agents)}",
                    f"Grid    : {self.grid.rows}x{self.grid.cols}",
            ):
                surf = font_s.render(line, True, PYGAME_TEXT)
                screen.blit(surf, (sx, sy));
                sy += 16
            sy += 8

            # legend
            screen.blit(font_m.render("Strategies", True, PYGAME_TEXT), (sx, sy))
            sy += 18
            seen = {}
            for agent in self.agents:
                lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
                seen[lbl] = agent_color[id(agent)]
            for lbl, col in seen.items():
                pygame.draw.rect(screen, col, (sx, sy + 2, 12, 12),
                                 border_radius=3)
                surf = font_s.render(lbl, True, PYGAME_TEXT)
                screen.blit(surf, (sx + 17, sy));
                sy += 17

            sy += 10
            screen.blit(font_s.render("Press Q to quit", True, (130, 130, 130)),
                        (sx, WIN_H - 20))

            pygame.display.flip()

        # ── main loop ───────────────────────────────────────────────
        running = True
        for ep in range(self.episodes):
            if not running:
                break
            self.current_episode = ep + 1

            # run episode tick-by-tick so we can render
            ep_strat_coop = defaultdict(int)
            ep_strat_total = defaultdict(int)

            for t in range(self.ticks_per_episode):
                # event pump
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False;
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False;
                            break
                if not running:
                    break

                sc, st = self._run_tick()
                for k in sc: ep_strat_coop[k] += sc[k]
                for k in st: ep_strat_total[k] += st[k]

                if t % skip == 0:
                    draw_frame(ep + 1, t + 1)
                    clock.tick(fps)

            # end-of-episode bookkeeping (same as run_episode)
            strat_scores = defaultdict(list)
            for agent in self.agents:
                lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
                strat_scores[lbl].append(agent.score)
            for lbl, sc in strat_scores.items():
                self.ep_avg_score[lbl].append(sum(sc) / len(sc))
            for lbl in ep_strat_total:
                r = (ep_strat_coop[lbl] / ep_strat_total[lbl]
                     if ep_strat_total[lbl] else 0)
                self.ep_coop_ratio[lbl].append(r)
            for agent in self.agents:
                if isinstance(agent, (Learner, Learner_Extended_History)):
                    self.ep_epsilon.append(agent.epsilon);
                    break
            else:
                self.ep_epsilon.append(None)

            if (ep + 1) % print_every == 0:
                self._print_episode_stats()

            if ep < self.episodes - 1 and running:
                for agent in self.agents:
                    if isinstance(agent, (Learner, Learner_Extended_History)):
                        agent.decay_epsilon()
                self.reset_episode()

        pygame.quit()
        self._print_final_summary()

    # ── logging ────────────────────────────────────────────────────
    def _print_episode_stats(self):
        print(f"\n===== EPISODE {self.current_episode} =====")
        for lbl in self.ep_avg_score:
            sc = self.ep_avg_score[lbl][-1]
            cr = self.ep_coop_ratio.get(lbl, [0])[-1]
            print(f"  {lbl:12s}  avg_score={sc:7.1f}  coop={cr:.2f}")
        eps = self.ep_epsilon[-1] if self.ep_epsilon else None
        if eps is not None:
            print(f"  epsilon = {eps:.4f}")

    def _print_final_summary(self):
        print("\n===== FINAL SUMMARY =====")
        w = min(100, self.episodes)
        for lbl in self.ep_avg_score:
            sc = self.ep_avg_score[lbl]
            cr = self.ep_coop_ratio.get(lbl, [])
            print(f"  {lbl:12s}  "
                  f"avg_score (last {w} ep): {sum(sc[-w:]) / w:7.1f}  "
                  f"coop (last {w} ep): {sum(cr[-w:]) / w:.3f}")

    # ── plot ───────────────────────────────────────────────────────
    def plot(self, smooth_window=50, save_path=None):
        """Same layout as Simulation.plot() — one image per chart in a subfolder."""
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
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        lbl_color = {lbl: FIXED_COLORS.get(lbl, prop_cycle[i % len(prop_cycle)])
                     for i, lbl in enumerate(self.ep_avg_score)}

        # 1. Global coop ratio per tick
        fig, ax = plt.subplots(figsize=(7, 4))
        raw = self.tick_coop_ratio
        x_sm, sm = smooth(raw, smooth_window)
        ax.plot(list(range(len(raw))), raw, alpha=0.12,
                color="steelblue", linewidth=0.6)
        ax.plot(x_sm, sm, color="steelblue", linewidth=1.8,
                label=f"smoothed (w={smooth_window})")
        ax.set_xlabel("Tick");
        ax.set_ylabel("Cooperation ratio")
        ax.set_title("Global cooperation ratio (per tick)")
        ax.set_ylim(-0.05, 1.05);
        ax.legend(fontsize=8)
        fig.tight_layout();
        save_or_show(fig, "1_global_coop")

        # 2. Avg score per strategy
        fig, ax = plt.subplots(figsize=(7, 4))
        for lbl, scores in self.ep_avg_score.items():
            ax.plot(eps_x, scores, label=lbl,
                    color=lbl_color[lbl], linewidth=1.4)
        ax.set_xlabel("Episode");
        ax.set_ylabel("Avg score")
        ax.set_title("Average score per strategy")
        ax.legend(fontsize=8)
        fig.tight_layout();
        save_or_show(fig, "2_avg_score")

        # 3. Coop ratio per strategy
        fig, ax = plt.subplots(figsize=(7, 4))
        w2 = max(1, smooth_window // 5)
        for lbl, crs in self.ep_coop_ratio.items():
            x_s, sm_crs = smooth(crs, w2)
            ax.plot(x_s, sm_crs, label=lbl,
                    color=lbl_color[lbl], linewidth=1.4)
        ax.set_xlabel("Episode");
        ax.set_ylabel("Cooperation ratio")
        ax.set_title(f"Cooperation ratio per strategy (smoothed w={w2})")
        ax.set_ylim(-0.05, 1.05);
        ax.legend(fontsize=8)
        fig.tight_layout();
        save_or_show(fig, "3_coop_ratio")

        # 4. Epsilon
        fig, ax = plt.subplots(figsize=(7, 4))
        eps_vals = [e for e in self.ep_epsilon if e is not None]
        if eps_vals:
            ax.plot(range(1, len(eps_vals) + 1), eps_vals,
                    color="darkorange", linewidth=1.5)
            ax.set_xlabel("Episode");
            ax.set_ylabel("Epsilon")
            ax.set_title("Learner exploration rate (ε)")
            ax.set_ylim(-0.02, max(eps_vals) * 1.05)
        else:
            ax.text(0.5, 0.5, "No Learner in simulation",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Epsilon (N/A)")
        fig.tight_layout();
        save_or_show(fig, "4_epsilon")

        # 5. Learner final policy by opponent type (separate panels per learner type)
        pid_to_label = {a.id: STRATEGY_LABEL.get(type(a), type(a).__name__)
                        for a in self.agents}

        pref_learner = defaultdict(lambda: [0, 0])
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

        has_learner = bool(pref_learner)
        has_learner_ext = bool(pref_learner_ext)
        n_panels = (1 if has_learner else 0) + (1 if has_learner_ext else 0)

        if n_panels == 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No Learner data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Learner policy by opponent type (N/A)")
            fig.tight_layout();
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
                ax_.bar([x - width / 2 for x in x_pos],
                        [pref_dict[t][0] for t in types], width,
                        label="Prefer C", color="steelblue")
                ax_.bar([x + width / 2 for x in x_pos],
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

            fig.tight_layout();
            save_or_show(fig, "5_learner_policy")

        # 6. Learner expected return by opponent type
        fig, ax = plt.subplots(figsize=(7, 4))
        ev_by_type = defaultdict(list)
        for agent in self.agents:
            if not isinstance(agent, (Learner, Learner_Extended_History)):
                continue
            for pid, qt in agent.qtables.items():
                lbl = pid_to_label.get(pid, "?")
                if qt:
                    ev_by_type[lbl].append(max(max(v) for v in qt.values()))
        if ev_by_type:
            types = sorted(ev_by_type)
            means = [sum(ev_by_type[t]) / len(ev_by_type[t]) for t in types]
            colors_bar = [lbl_color.get(t, "gray") for t in types]
            ax.bar(types, means, color=colors_bar)
            ax.set_ylabel("Max Q-value (proxy for expected return)")
            ax.set_title("Learner expected return by opponent type")
        else:
            ax.text(0.5, 0.5, "No Learner data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Learner expected return (N/A)")
        fig.tight_layout();
        save_or_show(fig, "6_learner_return")

        # 7. Pheromone heatmap (final state)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(self.grid.pheromone, cmap="YlOrRd", origin="upper")
        # overlay agent positions
        type_marker = {
            "Cooperator": ("o", "#1f77b4"),
            "Defector": ("x", "#ff7f0e"),
            "TitForTat": ("s", "#2ca02c"),
            "Unforgiving": ("D", "#8c564b"),
            "Pheromones": ("^", "#e377c2"),
            "Learner": ("P", "#d62728"),
            "LearnerExt": ("*", "#9467bd"),
        }
        for agent in self.agents:
            lbl = STRATEGY_LABEL.get(type(agent), type(agent).__name__)
            marker, color = type_marker.get(lbl, ("o", "white"))
            ax.plot(agent.col, agent.row, marker, color=color,
                    markersize=4, markeredgewidth=0.5, markeredgecolor="black")
        plt.colorbar(im, ax=ax, label="Pheromone level")
        ax.set_title("Pheromone field + agent positions (final state)")
        ax.set_xlabel("Col");
        ax.set_ylabel("Row")
        # legend
        handles = [plt.Line2D([0], [0], marker=m, color="w",
                              markerfacecolor=c, markersize=7, label=l)
                   for l, (m, c) in type_marker.items()
                   if any(STRATEGY_LABEL.get(type(a)) == l for a in self.agents)]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
        fig.tight_layout();
        save_or_show(fig, "7_pheromone_map")
