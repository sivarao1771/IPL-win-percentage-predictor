/**
 * Input rules and error messages for POST /api/predict.
 *
 * The range messages must equal input_contract in ml/artifacts/model_meta.json;
 * contract.spec.ts fails if they drift apart. Type messages are API-only.
 */
export const MAX_TARGET = 300;
export const MAX_BALLS_BOWLED = 119;
export const MAX_WICKETS_LOST = 9;

export const MESSAGES = {
  target: {
    type: 'target must be an integer',
    range: `target must be between 1 and ${MAX_TARGET}`,
  },
  score: {
    type: 'score must be an integer',
    range: 'score must be between 0 and target - 1',
  },
  ballsBowled: {
    type: 'ballsBowled must be an integer',
    range: `ballsBowled must be between 0 and ${MAX_BALLS_BOWLED}`,
  },
  wicketsLost: {
    type: 'wicketsLost must be an integer',
    range: `wicketsLost must be between 0 and ${MAX_WICKETS_LOST}`,
  },
} as const;
