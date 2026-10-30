import { createHash } from 'crypto';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { Inject, Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { FEATURE_ORDER } from '../prediction/features';
import { ModelMeta } from './model-meta';

export const MODEL_DIR = Symbol('MODEL_DIR');

export interface LoadedModel {
  session: InferenceSession;
  meta: ModelMeta;
}

async function readRequired(path: string): Promise<Buffer> {
  try {
    return await readFile(path);
  } catch {
    throw new Error(`Model file not found: ${path}`);
  }
}

/**
 * Load and check the model. Throws (so the app refuses to start) if a file is
 * missing, the model file does not match the SHA-256 in model_meta.json, or the
 * model's inputs and outputs are not what the meta file describes.
 */
export async function loadModel(modelDir: string): Promise<LoadedModel> {
  const meta = JSON.parse((await readRequired(join(modelDir, 'model_meta.json'))).toString('utf8')) as ModelMeta;
  // Hash and load the same bytes, so the checked file is the one that runs.
  const bytes = await readRequired(join(modelDir, 'model.onnx'));

  const sha256 = createHash('sha256').update(bytes).digest('hex');
  if (sha256 !== meta.model_sha256) {
    throw new Error(`model.onnx SHA-256 ${sha256} does not match model_meta.json (${meta.model_sha256})`);
  }
  if (meta.input.feature_order.join(',') !== FEATURE_ORDER.join(',')) {
    throw new Error(`Feature order in model_meta.json (${meta.input.feature_order.join(', ')}) ` +
      `does not match the API (${FEATURE_ORDER.join(', ')})`);
  }

  const session = await InferenceSession.create(bytes);
  if (!session.inputNames.includes(meta.input.name)) {
    throw new Error(`Model input "${meta.input.name}" not found; model has ${session.inputNames.join(', ')}`);
  }
  if (!session.outputNames.includes(meta.output.name)) {
    throw new Error(`Model output "${meta.output.name}" not found; model has ${session.outputNames.join(', ')}`);
  }
  return { session, meta };
}

@Injectable()
export class ModelService implements OnModuleInit {
  private readonly logger = new Logger(ModelService.name);
  private model?: LoadedModel;

  constructor(@Inject(MODEL_DIR) private readonly modelDir: string) {}

  async onModuleInit(): Promise<void> {
    this.model = await loadModel(this.modelDir);
    this.logger.log(`Loaded model ${this.model.meta.model_version} from ${this.modelDir}`);
  }

  get version(): string {
    return this.loaded().meta.model_version;
  }

  /** Win probabilities for rows of features laid out back to back, in FEATURE_ORDER. */
  async predict(rows: Float32Array): Promise<number[]> {
    const { session, meta } = this.loaded();
    const width = FEATURE_ORDER.length;
    if (rows.length === 0 || rows.length % width !== 0) {
      throw new RangeError(`Expected a multiple of ${width} values, got ${rows.length}`);
    }
    const input = new Tensor('float32', rows, [rows.length / width, width]);
    const output = await session.run({ [meta.input.name]: input }, [meta.output.name]);
    const probabilities = output[meta.output.name].data as Float32Array;
    const columns = probabilities.length / (rows.length / width);
    return Array.from({ length: rows.length / width },
      (_, i) => probabilities[i * columns + meta.output.win_probability_column]);
  }

  private loaded(): LoadedModel {
    if (!this.model) {
      throw new Error('Model not loaded');
    }
    return this.model;
  }
}
