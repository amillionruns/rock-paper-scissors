# Rock paper scissors, a million matches at a time

The code and data behind the A Million Runs video **We Played Rock Paper Scissors 1,000,000 Times**.

- Watch the video: https://www.youtube.com/watch?v=K3r9Z0-4K2A
- Play against Blank Sheet in your browser: https://amillionruns.com/rock-paper-scissors

Rock paper scissors is supposed to be pure luck. For every pairing below, we played a million matches, a thousand rounds each. Against a truly random player, every strategy we tried came out dead even. Against players with human habits, a simple bot that knows those habits won 41% of rounds and lost just 27%.

## Run it yourself

You need Python 3 and NumPy.

```
pip install numpy
python rps.py --quick
```

`--quick` plays 2,000 matches per pairing and prints a short summary in about a second. `python rps.py` plays the full million per pairing and writes `stats.json` and `extras.json`. That took about 4 minutes on our workstation.

The result files in this folder are the ones the video quotes. Before publishing, we re-ran `python rps.py` with this exact file (Python 3.12, NumPy 2.5) and every number came out identical. Every pairing uses a fixed random seed, so you get the same numbers we did. The settings are at the top of `rps.py`: change them and see what breaks.

| Key in `stats.json` | What it is |
|---|---|
| `luck` | Two random players, a million matches |
| `rock_vs_random`, `cheat_sheet_vs_random`, `blank_sheet_vs_random` | Three strategies against a random player |
| `human_vs_human` | The human model against itself |
| `cheat_sheet_vs_human`, `blank_sheet_vs_human` | The two bots against the human model |
| `habit_sweep` | Cheat Sheet against weaker and stronger versions of the human habits (100,000 matches of 100 rounds per setting) |

Scores count wins minus losses for the first player named. `snapshots` says how often that player was ahead, level or behind after 10, 30, 100, 300 and 1,000 rounds. `extras.json` holds the data for the charts in the video: final-score histograms, round-by-round win rates, 100 sample matches per pairing and the two throw-by-throw logs.

## The players

| Player | How it throws |
|---|---|
| Random | Uniformly at random, like a three-sided die |
| Rock-only | Rock, every single time |
| Human model | After its own win, tie or loss, it stays with its last throw or switches, with the chances in the table below |
| Cheat Sheet | Knows the human habits in advance. From your last throw and whether you won, tied or lost, it predicts your most likely next throw and plays what beats it. In words: if you just won or tied, it throws what beats your last throw; if you just lost, it throws your last throw. First throw: paper. |
| Blank Sheet | Knows nothing at the start. It keeps a tally of what you changed to after a win, a tie and a loss, bets you'll make your most common change again, and plays what beats that. First throw: random. |

"Switch up" means switching to the throw that beats your last one (rock → paper → scissors → rock). "Switch down" is the other way.

## The human model

Wang, Xu & Zhou (2014) had 360 students at Zhejiang University play 300 rounds in groups of six, against a randomly chosen opponent each round, for real cash. The paper shows how people changed their throw after a win, a tie and a loss only as bar charts (Fig. 3G–J), so we read the bars by eye (about ±0.01 each), averaged the four payoff settings where a tie paid clearly less than a win, and normalized each row.

| After your own... | Stay | Switch up | Switch down |
|---|---|---|---|
| Win | 48.0% | 23.0% | 29.0% |
| Tie | 38.5% | 36.0% | 25.5% |
| Loss | 33.7% | 27.5% | 38.8% |

The first throw uses the paper's overall shares (rock 36%, paper 33%, scissors 32%, normalized). The raw readings are in `stats.json` under `human_model.fig3_reads`.

## Where each number in the video comes from

Values here are rounded; the files have the full numbers.

| In the video | File | Value |
|---|---|---|
| Against a random player, Rock-only, Cheat Sheet and Blank Sheet all come out dead even | `stats.json` | `rock_vs_random`, `cheat_sheet_vs_random`, `blank_sheet_vs_random`: `win`, `tie`, `lose` all 0.333 |
| Two random players: about two thirds of matches end within a couple dozen points | `stats.json` | `luck.hist`: 666,501 of 1,000,000 between -25 and +25 |
| One match in 20 ends with someone more than 50 points ahead; the biggest blowout was 119 | `stats.json` | `luck.pct_abs_over_50` 0.0504, `luck.max_abs` 119 |
| After ten rounds, somebody's ahead in 85% of matches | `stats.json` | `luck.short_matches.10`: ahead 0.424 + behind 0.425 |
| Luck alone swings a 1,000-round match by about 50 points either way (Billings' figure) | `stats.json` | `luck.score_sd` 25.8, so about 95% of matches end within ±50.6 |
| Winners repeated their throw about half the time | `stats.json` | `human_model.by_last_outcome.win.stay` 0.48 |
| Cheat Sheet wins 41% of rounds and loses 27% | `stats.json` | `cheat_sheet_vs_human`: `win` 0.412, `lose` 0.273 |
| Ahead in 63% of matches after 10 rounds, 95% after 100, virtually all after 300 | `stats.json` | `cheat_sheet_vs_human.snapshots`: 0.632, 0.954, 0.999 |
| More than 50 points ahead in all but 170 of a million matches | `stats.json` | `cheat_sheet_vs_human.pct_over_50` 0.99983 |
| About 140 points over a thousand rounds, give or take a couple dozen | `stats.json` | `cheat_sheet_vs_human.score_mean` 139.6, `score_sd` 24.8 |
| Blank Sheet: basically guessing for ten rounds, ahead in about 3 of 4 matches after 100, 94% after 300, then almost as good as Cheat Sheet | `stats.json` | `blank_sheet_vs_human.by_round` 1-10: 0.344 / 0.328; `snapshots` 100: 0.737, 300: 0.943; `by_round` 301-1000: 0.407 / 0.294 |
| Habits at half strength: 37% wins, 30% losses, ahead in about 4 of 5 matches after 100 rounds; quarter strength: about 2 of 3; zero: the edge all but vanishes | `stats.json` | `habit_sweep` 0.5: 0.374 / 0.303, ahead 0.790; 0.25: ahead 0.648; 0: 0.334 / 0.333 |
| The two moves: about 41 wins for every 27 losses | `stats.json` | `cheat_sheet_vs_human` (the two moves are Cheat Sheet's rules) |

## What this model leaves out

- The human model comes from one experiment, read off charts. Real people vary, and some are much harder to read.
- It only reacts to its own last throw and whether it won. Real people also have favorite throws, streaks and moods, and they adapt when they notice they're being read.
- In the experiment, the students played each other and their habits roughly cancelled out: nobody made much money from them. Our numbers are for a bot that exploits those habits.
- Against a truly random player, nothing in here wins in the long run.

## The 1999 tournament

The video's history comes from Darse Billings' pages for the First International RoShamBo Programming Competition (University of Alberta, 1999). 45 programs plus 10 deliberately weak "dummy bots" played 1,000-round matches; Billings put the noise of a single match at about ±50.6 points, so the main event was run 25 times. Dan Egnor's Iocaine Powder won all 25. A purely random entry, Meta-Meta-Random, drew every match and finished 27th of 55. None of that is simulated here.

## Sources

- Wang, Z., Xu, B. & Zhou, H.-J. (2014). Social cycling and conditional responses in the Rock-Paper-Scissors game. *Scientific Reports* 4, 5830. https://doi.org/10.1038/srep05830
- Billings, D. (1999). First International RoShamBo Programming Competition, results part 1 and part 2. University of Alberta (archived): https://web.archive.org/web/2001/http://www.cs.ualberta.ca/~darse/rsb-results1.html and https://web.archive.org/web/2001/http://www.cs.ualberta.ca/~darse/rsb-results2.html

## License

MIT, see `LICENSE`.
