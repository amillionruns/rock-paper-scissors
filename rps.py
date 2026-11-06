"""Rock paper scissors, a million matches at a time (A Million Runs).

Run:  python rps.py           full run: a million matches per pairing, writes stats.json and extras.json
      python rps.py --quick   2,000 matches per pairing, prints a summary, writes nothing

Throws: 0 rock, 1 paper, 2 scissors.  (a - b) % 3: 1 = a wins, 2 = a loses, 0 = tie.
Changes from one throw to the next: 0 stay, 1 "up" (rock -> paper -> scissors -> rock: switch to the throw
that beats your last one), 2 "down" (rock -> scissors -> paper -> rock).

Every player is vectorized: one object plays n matches side by side, one round at a time.
"""
import json, pathlib, sys, time
import numpy as np

QUICK = "--quick" in sys.argv
N = 2_000 if QUICK else 1_000_000   # matches per pairing
ROUNDS = 1_000                       # rounds per match, as in the 1999 RoShamBo competition
SNAPS = (10, 30, 100, 300, 1000)     # who is ahead after this many rounds?

# Wang, Xu & Zhou 2014 (Scientific Reports 4:5830), Fig. 3G-J, read off the charts by eye (about +/-0.01).
# Each triple is (down, stay, up) for incentive parameters a = 2, 4, 9, 100; a = 1.1 is left out because
# a tie paid almost as much as a win there. The hidden a = 4 "down" bar after a tie is filled in as 1 - stay - up.
FIG3 = {
    "W": [(0.245, 0.53, 0.225), (0.30, 0.50, 0.21), (0.30, 0.45, 0.26), (0.32, 0.45, 0.23)],
    "T": [(0.23, 0.43, 0.33), (0.22, 0.39, 0.39), (0.30, 0.36, 0.35), (0.27, 0.36, 0.37)],
    "L": [(0.365, 0.36, 0.28), (0.38, 0.34, 0.28), (0.41, 0.31, 0.28), (0.40, 0.34, 0.26)],
}
FIRST = np.array([0.36, 0.33, 0.32]) / 1.01   # overall throw shares in the paper, used for the first throw only


def row(groups):
    down, stay, up = np.mean(groups, axis=0)
    v = np.array([stay, up, down])             # change order: 0 stay, 1 up, 2 down
    return v / v.sum()


# HUMAN[outcome][change]: chance of each change after your own last outcome (0 tie, 1 win, 2 loss)
HUMAN = np.array([row(FIG3["T"]), row(FIG3["W"]), row(FIG3["L"])])
UNIFORM = np.full((3, 3), 1 / 3)


class Random:
    name = "Random"
    def __init__(self, rng, n): self.rng, self.n = rng, n
    def move(self): return self.rng.integers(0, 3, self.n, dtype=np.int8)
    def see(self, mine, yours): pass


class Rock:
    name = "Rock-only"
    def __init__(self, rng, n): self.n = n
    def move(self): return np.zeros(self.n, np.int8)
    def see(self, mine, yours): pass


class Human:
    """The human model: after a win, a tie or a loss, stay or switch with the chances read from the paper."""
    name = "human model"
    def __init__(self, rng, n, P=HUMAN):
        self.rng, self.n, self.c0, self.c1 = rng, n, P.cumsum(1)[:, 0], P.cumsum(1)[:, 1]
        self.my_last = self.my_last_outcome = None
    def move(self):
        if self.my_last is None:
            return self.rng.choice(3, self.n, p=FIRST).astype(np.int8)
        u = self.rng.random(self.n)
        change = (u > self.c0[self.my_last_outcome]).astype(np.int8) + (u > self.c1[self.my_last_outcome])
        return ((self.my_last + change) % 3).astype(np.int8)
    def see(self, mine, yours):
        self.my_last, self.my_last_outcome = mine, ((mine - yours) % 3).astype(np.int8)


class CheatSheet:
    """Knows the human habits in advance. Guesses your most likely next throw and plays the counter.
    In words: if you just won, it throws what beats your last throw; if you just lost, it throws your last throw."""
    name = "Cheat Sheet"
    def __init__(self, rng, n, P=HUMAN):
        self.rng, self.n, self.best = rng, n, P.argmax(1).astype(np.int8)
        self.your_last = self.your_last_outcome = None
    def move(self):
        if self.your_last is None:
            return np.ones(self.n, np.int8)                   # first throw: paper (people open with rock most)
        guess = (self.your_last + self.best[self.your_last_outcome]) % 3
        return ((guess + 1) % 3).astype(np.int8)              # beat it
    def see(self, mine, yours):
        self.your_last, self.your_last_outcome = yours, ((yours - mine) % 3).astype(np.int8)


class BlankSheet:
    """Starts knowing nothing. Writes down what you changed to after a win, a tie and a loss,
    and bets that you will do your most common change again."""
    name = "Blank Sheet"
    def __init__(self, rng, n):
        self.rng, self.n, self.rows = rng, n, np.arange(n)
        # tally[outcome][change]: what you did after a tie (0), a win (1), a loss (2).
        # Starts at 1 plus a little noise, so ties between columns are broken at random.
        self.tally = 1 + 0.01 * rng.random((n, 3, 3), dtype=np.float32)
        self.your_last = self.your_last_outcome = None
    def move(self):
        if self.your_last is None:
            return self.rng.integers(0, 3, self.n, dtype=np.int8)
        change = self.tally[self.rows, self.your_last_outcome].argmax(1)
        guess = (self.your_last + change) % 3
        return ((guess + 1) % 3).astype(np.int8)              # beat it
    def see(self, mine, yours):
        if self.your_last is not None:
            self.tally[self.rows, self.your_last_outcome, (yours - self.your_last) % 3] += 1
        self.your_last, self.your_last_outcome = yours, ((yours - mine) % 3).astype(np.int8)


HIST_EDGES = np.arange(-125, 326, 10)   # final-score histogram, 10-point bins (same cut as luck.hist)


def play(a, b, n, rounds=ROUNDS, snaps=SNAPS, curve=False):
    """n matches of a vs b. Score = a's wins minus a's losses."""
    score = np.zeros(n, np.int32)
    win = lose = 0
    out = {"snapshots": {}}
    per_round = []
    for r in range(1, rounds + 1):
        ma, mb = a.move(), b.move()
        o = (ma - mb) % 3
        w, l = o == 1, o == 2
        score += w
        score -= l
        win += int(w.sum()); lose += int(l.sum())
        if curve:
            per_round.append((float(w.mean()), float(l.mean()), float((score > 0).mean()), float((score < 0).mean())))
        a.see(ma, mb); b.see(mb, ma)
        if r in snaps:
            out["snapshots"][r] = {"ahead": float((score > 0).mean()), "level": float((score == 0).mean()),
                                   "behind": float((score < 0).mean()), "mean": float(score.mean())}
    total = n * rounds
    out.update({"players": f"{a.name} vs {b.name}", "matches": n, "rounds": rounds,
                "win": win / total, "lose": lose / total, "tie": 1 - (win + lose) / total,
                "score_mean": float(score.mean()), "score_sd": float(score.std()),
                "pct_over_50": float((score > 50).mean()), "pct_under_minus_50": float((score < -50).mean())})
    if curve:
        blocks = [(1, 10), (11, 50), (51, 100), (101, 300), (301, 1000)]
        out["by_round"] = [{"rounds": f"{lo}-{hi}", "win": float(np.mean([per_round[i][0] for i in range(lo - 1, hi)])),
                            "lose": float(np.mean([per_round[i][1] for i in range(lo - 1, hi)]))} for lo, hi in blocks]
        # these two go to extras.json, not stats.json
        out["_per_round"] = {k: [round(x[i], 5) for x in per_round] for i, k in enumerate(["win", "lose", "ahead", "behind"])}
        out["_final_hist"] = {"edges": HIST_EDGES.tolist(), "counts": np.histogram(score, bins=HIST_EDGES)[0].tolist(),
                              "min": int(score.min()), "max": int(score.max()), "at_or_below_50": int((score <= 50).sum())}
    return out


def traces(make, seed, n=100):
    """n matches, running score after every round (the little tables in the video)."""
    rng = np.random.default_rng(seed)
    a, b = make(rng, n)
    score, rows = np.zeros(n, np.int32), []
    for _ in range(ROUNDS):
        ma, mb = a.move(), b.move()
        o = (ma - mb) % 3
        score += (o == 1).astype(np.int32) - (o == 2)
        rows.append(score.copy())
        a.see(ma, mb); b.see(mb, ma)
    return np.array(rows).T.tolist()


def log(make, seed, rounds):
    """Throw-by-throw record of one match (shown in the video): a's throw, b's throw, a's result (1 win, 2 loss, 0 tie)."""
    rng = np.random.default_rng(seed)
    a, b = make(rng, 1)
    rows = []
    for _ in range(rounds):
        ma, mb = a.move(), b.move()
        rows.append({"a": "RPS"[int(ma[0])], "b": "RPS"[int(mb[0])], "a_result": int((ma[0] - mb[0]) % 3)})
        a.see(ma, mb); b.see(mb, ma)
    return rows


def luck(seed):
    """Two random players. Each round is a win, tie or loss with 1/3 each, so we can draw whole matches at once."""
    rng = np.random.default_rng(seed)
    wtl = rng.multinomial(ROUNDS, [1 / 3] * 3, size=N)
    score = wtl[:, 0] - wtl[:, 2]
    edges = np.arange(-125, 126, 10)
    hist, _ = np.histogram(score, bins=edges)
    short = {}
    for k in SNAPS:
        s = rng.multinomial(k, [1 / 3] * 3, size=N)
        d = s[:, 0] - s[:, 2]
        short[k] = {"ahead": float((d > 0).mean()), "level": float((d == 0).mean()), "behind": float((d < 0).mean())}
    return {"players": "Random vs Random", "matches": N, "rounds": ROUNDS, "seed": seed,
            "score_mean": float(score.mean()), "score_sd": float(score.std()),
            "pct_abs_over_50": float((np.abs(score) > 50).mean()), "pct_over_50": float((score > 50).mean()),
            "max_abs": int(np.abs(score).max()), "hist_edges": edges.tolist(), "hist": hist.tolist(),
            "short_matches": short}


def main():
    t0 = time.time()
    stats = {"note": "A Million Runs: rock paper scissors. Human model read off Wang, Xu & Zhou 2014, Fig. 3G-J.", "N": N, "rounds": ROUNDS,
             "human_model": {"by_last_outcome": {k: dict(zip(["stay", "up", "down"], HUMAN[i].round(4).tolist()))
                                                 for i, k in enumerate(["tie", "win", "loss"])},
                             "first_move": dict(zip("RPS", FIRST.round(4).tolist())), "fig3_reads": FIG3}}
    stats["luck"] = luck(1)
    print("luck", round(time.time() - t0, 1), "s", flush=True)

    runs = [
        ("rock_vs_random", lambda r: (Rock(r, N), Random(r, N)), 11, False),
        ("cheat_sheet_vs_random", lambda r: (CheatSheet(r, N), Random(r, N)), 12, False),
        ("blank_sheet_vs_random", lambda r: (BlankSheet(r, N), Random(r, N)), 13, False),
        ("human_vs_human", lambda r: (Human(r, N), Human(r, N)), 21, False),
        ("cheat_sheet_vs_human", lambda r: (CheatSheet(r, N), Human(r, N)), 22, True),
        ("blank_sheet_vs_human", lambda r: (BlankSheet(r, N), Human(r, N)), 23, True),
    ]
    for key, make, seed, curve in runs:
        rng = np.random.default_rng(seed)
        stats[key] = play(*make(rng), N, curve=curve) | {"seed": seed}
        print(key, round(time.time() - t0, 1), "s", flush=True)

    # Habit strength: 0 = fully random, 1 = the people in the paper, 1.5 = even more predictable.
    # 100,000 matches of 100 rounds per setting, Cheat Sheet (tuned to the same habits) vs the human model.
    sweep = []
    for i, lam in enumerate([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]):
        P = UNIFORM + lam * (HUMAN - UNIFORM)
        n = 2_000 if QUICK else 100_000
        rng = np.random.default_rng(100 + i)
        res = play(CheatSheet(rng, n, P), Human(rng, n, P), n, rounds=100, snaps=(100,))
        sweep.append({"habit": lam, "matches": n, "rounds": 100, "win": res["win"], "lose": res["lose"],
                      "tie": res["tie"], "ahead_after_100": res["snapshots"][100]["ahead"]})
    stats["habit_sweep"] = sweep
    stats["seconds"] = round(time.time() - t0, 1)
    print("done", stats["seconds"], "s", flush=True)

    extras = {"note": "Chart and demo data for the video; see the top of rps.py", "final_hist": {}, "per_round": {}}
    for key in ["cheat_sheet_vs_human", "blank_sheet_vs_human"]:
        extras["final_hist"][key] = stats[key].pop("_final_hist")
        extras["per_round"][key] = stats[key].pop("_per_round")
    extras["traces"] = {
        "random_vs_random": {"seed": 31, "scores": traces(lambda r, n: (Random(r, n), Random(r, n)), 31)},
        "cheat_sheet_vs_human": {"seed": 32, "scores": traces(lambda r, n: (CheatSheet(r, n), Human(r, n)), 32)},
        "blank_sheet_vs_human": {"seed": 33, "scores": traces(lambda r, n: (BlankSheet(r, n), Human(r, n)), 33)},
    }
    extras["logs"] = {
        "cheat_sheet_vs_human": {"seed": 41, "a": "Cheat Sheet", "b": "human model", "rounds": log(lambda r, n: (CheatSheet(r, n), Human(r, n)), 41, 60)},
        "blank_sheet_vs_human": {"seed": 42, "a": "Blank Sheet", "b": "human model", "rounds": log(lambda r, n: (BlankSheet(r, n), Human(r, n)), 42, 300)},
    }

    if QUICK:
        for k in ["cheat_sheet_vs_human", "blank_sheet_vs_human", "human_vs_human", "cheat_sheet_vs_random"]:
            print(k, {x: round(stats[k][x], 4) for x in ["win", "tie", "lose", "score_mean"]})
    else:
        here = pathlib.Path(__file__)
        here.with_name("stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        here.with_name("extras.json").write_text(json.dumps(extras, separators=(",", ":")), encoding="utf-8")
        print("wrote stats.json and extras.json")


if __name__ == "__main__":
    main()
