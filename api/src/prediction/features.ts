/**
 * Chase features: a TypeScript copy of ml/features.py.
 *
 * The formulas and the order of operations must stay identical to Python, so
 * both sides produce bit-identical float64 values before converting to float32.
 * The parity tests (features.spec.ts) check this against 200 cases from Python.
 */
export const TOTAL_BALLS = 120;
export const TOTAL_WICKETS = 10;

/** Column order of the model input. Must equal feature_order in model_meta.json. */
export const FEATURE_ORDER = ['runs_left', 'balls_left', 'wickets_left', 'runs_target', 'crr', 'rrr'] as const;

export interface MatchState {
  target: number;
  score: number;
  ballsBowled: number;
  wicketsLost: number;
}

export interface ChaseFeatures {
  runsLeft: number;
  ballsLeft: number;
  wicketsLeft: number;
  runsTarget: number;
  crr: number;
  rrr: number;
}

export function computeFeatures(state: MatchState): ChaseFeatures {
  const runsLeft = state.target - state.score;
  const ballsLeft = TOTAL_BALLS - state.ballsBowled;
  // Validation guarantees an undecided chase; this guards against misuse.
  if (runsLeft < 1 || ballsLeft < 1) {
    throw new RangeError('computeFeatures needs an undecided chase (runs left and balls left >= 1)');
  }
  return {
    runsLeft,
    ballsLeft,
    wicketsLeft: TOTAL_WICKETS - state.wicketsLost,
    runsTarget: state.target,
    crr: state.ballsBowled > 0 ? (state.score * 6) / state.ballsBowled : 0,
    rrr: (runsLeft * 6) / ballsLeft,
  };
}

/** One model input row, in FEATURE_ORDER, as float32. */
export function toModelInput(f: ChaseFeatures): Float32Array {
  return Float32Array.from([f.runsLeft, f.ballsLeft, f.wicketsLeft, f.runsTarget, f.crr, f.rrr]);
}
