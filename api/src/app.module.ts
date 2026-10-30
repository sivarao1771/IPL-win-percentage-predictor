import { Module } from '@nestjs/common';
import { HealthController } from './health/health.controller';
import { ModelModule } from './model/model.module';
import { PredictionModule } from './prediction/prediction.module';

@Module({
  imports: [ModelModule, PredictionModule],
  controllers: [HealthController],
})
export class AppModule {}
