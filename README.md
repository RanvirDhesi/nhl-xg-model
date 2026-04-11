# NHL Expected Goals (xG) Model

A machine learning model that predicts the probability of any NHL (National Hockey League) shot becoming a goal. Trained on **2,066,537 shots across 19 seasons (2007-2025)** of NHL data.

Built by [@SensRandy](https://twitter.com/SensRandy)

---

## What is xG?

xG stands for "expected goals." It's a number between 0 and 1 that represents how likely a shot is to go in, based on historical data.

A breakaway from 10 feet out might be assigned 0.35 (35% chance of scoring). A point shot from the blue line through traffic might be 0.02 (2% chance). Our model calculates this for every single shot using 52 different inputs.

**Why it matters:** If a team generates 3.5 xG in a game but only scores 1 goal, they were unlucky. They created enough high-quality chances to deserve 3 or 4 goals. xG tells you more about how well a team actually played than the final score does.

<p align="center">
  <img src="assets/shotmap_example.png" alt="Shot Map Example" width="600"/>
  <br>
  <em>Shot map from an Ottawa Senators game. Dot size represents chance quality. Stars mark goals scored.</em>
</p>

---

## Model Performance

| Metric | Value | What This Means |
|--------|-------|---------|
| **AUC (Area Under the ROC Curve)** | 0.786 | If you pick one random shot that scored and one that didn't, the model correctly identifies which one was more dangerous 78.6% of the time. A coin flip would score 0.50. A perfect model would score 1.00. |
| **Brier Score** | 0.056 | Measures how close our predicted probabilities are to what actually happened. Lower is better. 0.056 indicates strong calibration. |
| **Training Data** | 2,066,537 shots | Covers 19 full NHL seasons from 2007 through 2025. |
| **Features** | 52 | The number of inputs the model evaluates for each shot. |

We validated the model using 5-fold cross-validation, meaning we split the data into 5 groups, trained on 4 and tested on the 5th, and repeated this 5 times. The AUC varied by only 0.0004 across folds, which indicates the model is extremely stable and not overfitting to any particular slice of data.

This performance is in line with published benchmarks from [MoneyPuck](https://moneypuck.com/) and other public xG models. The theoretical ceiling without NHL player tracking data (puck speed, goalie positioning, screen traffic) is roughly 0.80 to 0.81.

---

## How It Works

### Algorithm

The model uses **CatBoost**, a gradient-boosted decision tree algorithm. In simple terms, it builds 1,500 small decision trees in sequence. Each tree learns from the mistakes of the ones before it. The final prediction is the combined output of all 1,500 trees working together.

CatBoost was chosen over alternatives like XGBoost because it handles categorical inputs (such as shot type and player position) natively, without requiring manual encoding.

### Features (What the Model Evaluates)

For every shot, the model looks at 52 different inputs, grouped into the following categories:

#### Shot Location (6 features)
Where the shot was taken from. This includes distance and angle to the net, with corrections for arena reporting bias (different NHL arenas record shot coordinates slightly differently).

> Shots from within 20 feet score roughly 15% of the time. Shots from beyond 50 feet score around 2%.

#### Shot Type (1 feature)
The type of shot: wrist, slap, snap, backhand, tip-in, wrap-around, or deflection. Each type converts at a different rate.

> Deflections and tip-ins are the most dangerous shot types. Wrist shots are the most common.

#### Shot Context (6 features)
- **Rebound:** Was this a second-chance shot after a save? Rebounds score at 10.9%, well above the 6.7% league average.
- **Rush:** Was this shot taken during a transition play? Rush chances tend to be higher quality.
- **Empty net:** Was the opposing goalie pulled from the net?
- **Off-wing:** Was the shooter on their off-wing (e.g., a left-handed shooter on the right side)? This creates different shooting angles.

#### Game State (7 features)
The number of skaters on the ice for each team (5-on-5, power play, shorthanded), the period of play, whether the team is home or away, and whether it is a regular season or playoff game.

> Power play shots convert at roughly double the 5-on-5 rate because the shooting team has more time, more space, and better passing lanes.

#### Score State (2 features + 2 engineered)
Whether the shooting team is leading, trailing, or tied, along with the exact goal differential.

> Teams that are trailing tend to take more aggressive, higher-quality shots. Teams with a lead play more conservatively.

#### Last Event Context (6 features)
What happened immediately before the shot. This includes the distance, angle, speed, and type of the previous event on the play-by-play.

> A shot right after a turnover in the slot is a very different situation than one after a 30-second cycle along the boards.

#### Fatigue (8 features)
How long the shooter has been on the ice during the current shift, time elapsed since the last whistle, average shift length for both teams, and rest differential between the two teams.

> Players late in a shift are slower to release the puck and less accurate. Fresh legs generate better scoring chances.

#### Player Identity (2 features)
The shooter's position (center, left wing, right wing, defenseman) and their shooting hand (left or right).

> Defensemen take the majority of shots from long range. Forwards generate most of the high-danger chances close to the net.

#### Engineered Features (12 features)

We built 12 additional features on top of the raw data. Three of these are novel features based on hockey domain knowledge, and they are the ones that meaningfully improve the model. The other 9 are standard derivations (score differential, power play flags, fatigue ratios, etc.) that reformat existing columns into forms the model can learn from more effectively. We don't list those 9 individually because they are standard practice in any xG model. Combined, they account for about 5% of overall model importance.

The three novel features:

1. **Zone Danger.** We divided the ice surface into a grid of small zones and computed the historical goal rate from each zone across all 2 million+ shots. This captures spatial scoring patterns that raw distance and angle miss on their own. For example, shots from the left hash marks convert at a different rate than shots from the right circle at the same distance from the net. This is our single most important feature, accounting for 17.8% of model importance.

2. **Shooter Talent.** Each shooter's historical shooting percentage relative to the league average, adjusted using Bayesian shrinkage for players with small sample sizes. An elite scorer like Alex Ovechkin receives a positive adjustment. A fourth-line player with limited career data gets pulled back toward the league average so the model doesn't overreact to small samples.

3. **Goalie Quality.** Each goalie's historical save percentage relative to the league average, with the same small-sample adjustment. The identity of the goalie in net matters. Facing a Vezina Trophy caliber starter is a very different situation than facing a backup or an AHL (American Hockey League) callup.

---

## Feature Importance

The top 10 features account for roughly 70% of what the model learns:

```
Zone Danger .......................... 17.8%
Shot Distance ........................ 10.7%
Arena-Adjusted Distance .............. 8.3%
Shot Type ............................ 7.4%
Time Since Last Event ................ 5.5%
Goalie Quality ....................... 4.9%
Shooter Talent ....................... 4.7%
Lateral Position (Y-axis) ........... 4.3%
Offensive Zone Depth (X-axis) ....... 3.6%
Shot Angle ........................... 2.9%
```

The remaining 42 features contribute the other 30%. Each one is individually small but they are collectively meaningful.

---

## Training Process

1. **Data Collection.** 19 seasons of shot-level data sourced from MoneyPuck (2007 through 2025), totaling 2,066,537 shots and 139,272 goals at a 6.74% overall goal rate.

2. **Feature Engineering.** 40 features selected from the raw data, plus 12 custom features built from hockey domain knowledge, for a total of 52.

3. **Cross-Validation.** 5-fold stratified cross-validation. The full dataset is split into 5 equal groups. The model trains on 4 groups (~1.65 million shots) and validates on the remaining group (~413,000 shots). This is repeated 5 times so every shot is used for validation exactly once.

4. **Final Model.** After validation confirms performance, a final model is trained on all 2 million+ shots for production use.

5. **Leakage Detection.** During development, we discovered and removed a feature called `shotGeneratedRebound` that was encoding the outcome we were trying to predict. A shot that "generated a rebound" was, by definition, a shot that had been saved. We confirmed this: 0 goals out of 8,534 shots where this flag was present. Including it would have inflated our metrics dishonestly, so we removed it.

---

## Applications

This model powers the analytics content on [@SensRandy](https://twitter.com/SensRandy):

- **Shot Maps.** A visual display of every shot in a game, with each dot sized according to its goal probability.
- **xG Flow Charts.** Cumulative expected goals plotted over the course of a game, showing which team was creating more dangerous chances and when.
- **Post-Game Verdicts.** Analysis of whether a team "deserved to win" based on the quality of chances they created vs. what they allowed.
- **Player Analysis.** Identifying which players are outperforming or underperforming relative to their expected output.

---

## Limitations

- **No tracking data.** We do not have access to player skating speed, puck velocity, goalie positioning, or screening information. The NHL captures this data internally but does not make it publicly available in bulk.
- **No pre-shot passing data.** Cross-ice passes and one-timer shots dramatically increase goal probability, but this information is not fully captured in the publicly available feature set.
- **Historical bias.** The game has changed meaningfully over 19 seasons due to rule changes, goaltending equipment regulations, and evolving playing styles. Older seasons may not perfectly represent how modern hockey is played.
- **Public data only.** This model is built on the same MoneyPuck data that is available to anyone. NHL teams with proprietary tracking data and video analysis can build meaningfully better models.

---

## How Does This Compare?

| Model | AUC | Notes |
|-------|-----|------|
| Random guess | 0.500 | Baseline. No predictive ability. |
| Shot distance only | ~0.680 | A single-feature model using just how far the shot was from the net. |
| **This model** | **0.786** | **52 features trained on 2 million shots.** |
| MoneyPuck (industry standard) | ~0.78-0.80 | 124 features. Maintained by Peter Tanner. Widely used in hockey analytics. |
| Theoretical ceiling (public data) | ~0.80-0.81 | Reaching higher would require player/puck tracking data that is not publicly available. |

---

## Tech Stack

- **Model:** CatBoost gradient-boosted classifier
- **Training Data:** MoneyPuck shot-level CSV files (2007-2025)
- **Live Scoring:** NHL API play-by-play data, scored in real time after each game
- **Visualization:** Matplotlib for shot maps and xG flow charts
- **Infrastructure:** AWS (Amazon Web Services)

---

## Contact

- Twitter: [@SensRandy](https://twitter.com/SensRandy) (Ottawa Senators analytics)

Built with data from [MoneyPuck.com](https://moneypuck.com/) and the [NHL API](https://api-web.nhle.com/).
