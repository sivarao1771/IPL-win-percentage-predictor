import { Module } from '@nestjs/common';
import { DEFAULT_MODEL_DIR } from '../config/paths';
import { MODEL_DIR, ModelService } from './model.service';

@Module({
  providers: [
    { provide: MODEL_DIR, useFactory: () => process.env.MODEL_DIR ?? DEFAULT_MODEL_DIR },
    ModelService,
  ],
  exports: [ModelService],
})
export class ModelModule {}
