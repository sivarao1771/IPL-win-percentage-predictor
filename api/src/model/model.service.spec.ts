import { cpSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { meta, parity } from '../../test/parity';
import { DEFAULT_MODEL_DIR } from '../config/paths';
import { FEATURE_ORDER } from '../prediction/features';
import { loadModel, ModelService } from './model.service';

describe('ModelService', () => {
  let service: ModelService;

  beforeAll(async () => {
    service = new ModelService(DEFAULT_MODEL_DIR);
    await service.onModuleInit();
  });

  it('reports the version from model_meta.json', () => {
    expect(service.version).toBe(meta.model_version);
  });

  it('matches the Python probabilities for all parity cases (one batch)', async () => {
    const rows = Float32Array.from(parity.cases.flatMap((c) => FEATURE_ORDER.map((name) => c.features[name])));
    const predicted = await service.predict(rows);
    const gaps = predicted.map((p, i) => Math.abs(p - parity.cases[i].winProbability));
    expect(Math.max(...gaps)).toBeLessThanOrEqual(parity.tolerance);
  });

  it('gives the same answer for a single row as in a batch', async () => {
    const c = parity.cases[0];
    const [p] = await service.predict(Float32Array.from(FEATURE_ORDER.map((name) => c.features[name])));
    expect(Math.abs(p - c.winProbability)).toBeLessThanOrEqual(parity.tolerance);
  });

  it('rejects input that is not whole rows of 6 features', async () => {
    await expect(service.predict(new Float32Array(5))).rejects.toThrow(RangeError);
    await expect(service.predict(new Float32Array(0))).rejects.toThrow(RangeError);
  });
});

describe('loadModel refuses to start on bad model files', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'ipl-model-'));
    cpSync(DEFAULT_MODEL_DIR, dir, { recursive: true });
  });

  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  const editMeta = (change: (m: Record<string, any>) => void) => {
    const path = join(dir, 'model_meta.json');
    const m = JSON.parse(readFileSync(path, 'utf8'));
    change(m);
    writeFileSync(path, JSON.stringify(m));
  };

  it('loads the real files', async () => {
    await expect(loadModel(dir)).resolves.toHaveProperty('meta.model_version', meta.model_version);
  });

  it('fails when model.onnx is missing', async () => {
    unlinkSync(join(dir, 'model.onnx'));
    await expect(loadModel(dir)).rejects.toThrow(/Model file not found: .*model\.onnx/);
  });

  it('fails when model_meta.json is missing', async () => {
    unlinkSync(join(dir, 'model_meta.json'));
    await expect(loadModel(dir)).rejects.toThrow(/Model file not found: .*model_meta\.json/);
  });

  it('fails when the model file does not match its SHA-256', async () => {
    editMeta((m) => { m.model_sha256 = '0'.repeat(64); });
    await expect(loadModel(dir)).rejects.toThrow(/SHA-256 .* does not match/);
  });

  it('fails when the input name is wrong', async () => {
    editMeta((m) => { m.input.name = 'wrong_input'; });
    await expect(loadModel(dir)).rejects.toThrow(/Model input "wrong_input" not found/);
  });

  it('fails when the feature order differs from the API', async () => {
    editMeta((m) => { m.input.feature_order = [...m.input.feature_order].reverse(); });
    await expect(loadModel(dir)).rejects.toThrow(/Feature order/);
  });
});
