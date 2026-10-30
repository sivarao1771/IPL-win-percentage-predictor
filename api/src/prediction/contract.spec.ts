import { meta } from '../../test/parity';
import { MAX_BALLS_BOWLED, MAX_TARGET, MAX_WICKETS_LOST, MESSAGES } from './contract';

describe('input contract', () => {
  it('uses exactly the range messages from model_meta.json', () => {
    const fromMeta = Object.fromEntries(
      Object.entries(meta.input_contract).map(([field, rule]) => [field, rule.message]));
    const fromApi = Object.fromEntries(
      Object.entries(MESSAGES).map(([field, messages]) => [field, messages.range]));
    expect(fromApi).toEqual(fromMeta);
  });

  it('uses the limits from model_meta.json', () => {
    const c = meta.input_contract;
    expect([c.target.min, c.target.max]).toEqual([1, MAX_TARGET]);
    expect([c.score.min, c.score.max]).toEqual([0, 'target - 1']);
    expect([c.ballsBowled.min, c.ballsBowled.max]).toEqual([0, MAX_BALLS_BOWLED]);
    expect([c.wicketsLost.min, c.wicketsLost.max]).toEqual([0, MAX_WICKETS_LOST]);
  });
});
