import { Injectable } from '@nestjs/common';
import { ModelService } from '../model/model.service';
import { computeFeatures, MatchState, toModelInput } from './features';
import { PredictResponseDto } from './predict.dto';

@Injectable()
export class PredictionService {
  constructor(private readonly model: ModelService) {}

  async predict(state: MatchState): Promise<PredictResponseDto> {
    const features = computeFeatures(state);
    const [winProbability] = await this.model.predict(toModelInput(features));
    const { runsLeft, ballsLeft, wicketsLeft, crr, rrr } = features;
    return {
      winProbability,
      modelVersion: this.model.version,
      features: { runsLeft, ballsLeft, wicketsLeft, crr, rrr },
    };
  }
}
