import { readFileSync } from 'fs';
import { join } from 'path';
import { DEFAULT_MODEL_DIR } from '../src/config/paths';
import { ModelMeta } from '../src/model/model-meta';
import { MatchState } from '../src/prediction/features';

/** Test fixtures written by ml/export_onnx.py. */
export interface ParityCase {
  description: string;
  input: MatchState;
  features: Record<'runs_left' | 'balls_left' | 'wickets_left' | 'runs_target' | 'crr' | 'rrr', number>;
  winProbability: number;
}

export const parity = JSON.parse(readFileSync(join(DEFAULT_MODEL_DIR, 'parity_cases.json'), 'utf8')) as {
  tolerance: number;
  cases: ParityCase[];
};

export const meta = JSON.parse(readFileSync(join(DEFAULT_MODEL_DIR, 'model_meta.json'), 'utf8')) as ModelMeta;
