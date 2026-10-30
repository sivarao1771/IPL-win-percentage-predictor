import { Module } from '@nestjs/common';
import { ModelModule } from '../model/model.module';
import { PredictionController } from './prediction.controller';
import { PredictionService } from './prediction.service';

@Module({
  imports: [ModelModule],
  controllers: [PredictionController],
  providers: [PredictionService],
})
export class PredictionModule {}
