import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import request from 'supertest';
import { AppModule } from '../src/app.module';
import { configureApp } from '../src/app.setup';
import { meta, parity } from './parity';

const valid = { target: 180, score: 95, ballsBowled: 66, wicketsLost: 3 };

describe('API (end to end)', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] }).compile();
    app = moduleRef.createNestApplication({ logger: false });
    configureApp(app);
    await app.init();
  });

  afterAll(() => app.close());

  const predict = (body: unknown) => request(app.getHttpServer()).post('/api/predict').send(body as object);

  describe('POST /api/predict with valid input', () => {
    it('returns the probability, model version and derived features', async () => {
      const res = await predict(valid).expect(200);
      expect(res.body).toEqual({
        winProbability: expect.any(Number),
        modelVersion: meta.model_version,
        features: { runsLeft: 85, ballsLeft: 54, wicketsLeft: 7, crr: (95 * 6) / 66, rrr: (85 * 6) / 54 },
      });
      expect(res.body.winProbability).toBeGreaterThanOrEqual(0);
      expect(res.body.winProbability).toBeLessThanOrEqual(1);
    });

    const edgeCases = parity.cases.filter((c) => !c.description.startsWith('test row'));
    it.each(edgeCases.map((c) => [c.description, c] as const))(
      'matches Python over HTTP: %s', async (_, c) => {
        const res = await predict(c.input).expect(200);
        expect(Math.abs(res.body.winProbability - c.winProbability)).toBeLessThanOrEqual(parity.tolerance);
      });

    it.each([
      ['lowest target', { ...valid, target: 1, score: 0 }],
      ['highest target', { ...valid, target: 300 }],
      ['score = target - 1', { ...valid, score: 179 }],
      ['score 0', { ...valid, score: 0 }],
      ['ballsBowled 0', { ...valid, ballsBowled: 0 }],
      ['ballsBowled 119', { ...valid, ballsBowled: 119 }],
      ['wicketsLost 0', { ...valid, wicketsLost: 0 }],
      ['wicketsLost 9', { ...valid, wicketsLost: 9 }],
    ])('accepts the boundary: %s', async (_, body) => {
      await predict(body).expect(200);
    });
  });

  describe('POST /api/predict with invalid input returns 400 and the exact message', () => {
    it.each([
      ['target 0', { ...valid, target: 0 }, ['target must be between 1 and 300']],
      ['target 301', { ...valid, target: 301 }, ['target must be between 1 and 300']],
      ['score -1', { ...valid, score: -1 }, ['score must be between 0 and target - 1']],
      ['score = target', { ...valid, score: 180 }, ['score must be between 0 and target - 1']],
      ['score > target', { ...valid, score: 250 }, ['score must be between 0 and target - 1']],
      ['ballsBowled -1', { ...valid, ballsBowled: -1 }, ['ballsBowled must be between 0 and 119']],
      ['ballsBowled 120', { ...valid, ballsBowled: 120 }, ['ballsBowled must be between 0 and 119']],
      ['wicketsLost -1', { ...valid, wicketsLost: -1 }, ['wicketsLost must be between 0 and 9']],
      ['wicketsLost 10', { ...valid, wicketsLost: 10 }, ['wicketsLost must be between 0 and 9']],
      ['decimal', { ...valid, target: 180.5 }, ['target must be an integer']],
      ['number as a string', { ...valid, score: '95' }, ['score must be an integer']],
      ['null', { ...valid, ballsBowled: null }, ['ballsBowled must be an integer']],
      ['boolean', { ...valid, wicketsLost: true }, ['wicketsLost must be an integer']],
      ['missing field', { target: 180, score: 95, ballsBowled: 66 }, ['wicketsLost must be an integer']],
      ['two bad fields', { ...valid, target: 0, wicketsLost: 'x' },
        ['target must be between 1 and 300', 'wicketsLost must be an integer']],
      ['unknown field', { ...valid, battingTeam: 'Mumbai Indians' }, ['property battingTeam should not exist']],
      ['empty body', {}, ['target must be an integer', 'score must be an integer',
        'ballsBowled must be an integer', 'wicketsLost must be an integer']],
    ])('%s', async (_, body, messages) => {
      const res = await predict(body).expect(400);
      expect(res.body).toEqual({ statusCode: 400, error: 'Bad Request', message: messages });
    });

    it('reports an invalid target, not a score error, when both depend on target', async () => {
      const res = await predict({ ...valid, target: 'abc' }).expect(400);
      expect(res.body.message).toEqual(['target must be an integer']);
    });
  });

  describe('other routes', () => {
    it('GET /api/health reports the loaded model', async () => {
      const res = await request(app.getHttpServer()).get('/api/health').expect(200);
      expect(res.body).toEqual({ status: 'ok', modelVersion: meta.model_version });
    });

    it('serves the Swagger UI and the OpenAPI document', async () => {
      await request(app.getHttpServer()).get('/docs').expect(200).expect('Content-Type', /html/);
      const res = await request(app.getHttpServer()).get('/docs-json').expect(200);
      expect(Object.keys(res.body.paths)).toEqual(expect.arrayContaining(['/api/predict', '/api/health']));
      expect(res.body.components.schemas.PredictRequestDto.properties.target).toMatchObject({ minimum: 1, maximum: 300 });
    });

    it('returns 404 for unknown routes', async () => {
      await request(app.getHttpServer()).get('/api/unknown').expect(404);
    });
  });
});
