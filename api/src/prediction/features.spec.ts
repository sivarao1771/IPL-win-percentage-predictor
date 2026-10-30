import { parity } from '../../test/parity';
import { computeFeatures, toModelInput } from './features';

describe('computeFeatures', () => {
  it.each(parity.cases.map((c) => [c.description, c] as const))(
    'matches Python exactly: %s', (_, c) => {
      const f = computeFeatures(c.input);
      // toBe, not toBeCloseTo: same formulas in the same order give identical float64 values.
      expect(f.runsLeft).toBe(c.features.runs_left);
      expect(f.ballsLeft).toBe(c.features.balls_left);
      expect(f.wicketsLeft).toBe(c.features.wickets_left);
      expect(f.runsTarget).toBe(c.features.runs_target);
      expect(f.crr).toBe(c.features.crr);
      expect(f.rrr).toBe(c.features.rrr);
    });

  it('works out a chase by hand', () => {
    // Chasing 150, 60 for 2 after 48 balls: need 90 off 72.
    expect(computeFeatures({ target: 150, score: 60, ballsBowled: 48, wicketsLost: 2 })).toEqual({
      runsLeft: 90, ballsLeft: 72, wicketsLeft: 8, runsTarget: 150, crr: 7.5, rrr: 7.5,
    });
  });

  it('uses a current run rate of 0 before the first ball', () => {
    expect(computeFeatures({ target: 180, score: 0, ballsBowled: 0, wicketsLost: 0 }).crr).toBe(0);
  });

  it('refuses a chase that is already decided', () => {
    expect(() => computeFeatures({ target: 150, score: 150, ballsBowled: 60, wicketsLost: 2 })).toThrow(RangeError);
    expect(() => computeFeatures({ target: 150, score: 100, ballsBowled: 120, wicketsLost: 2 })).toThrow(RangeError);
  });
});

describe('toModelInput', () => {
  it('orders features like the model and converts to float32', () => {
    const f = computeFeatures({ target: 180, score: 95, ballsBowled: 66, wicketsLost: 3 });
    expect(Array.from(toModelInput(f))).toEqual(
      Array.from(Float32Array.from([85, 54, 7, 180, (95 * 6) / 66, (85 * 6) / 54])));
  });
});
