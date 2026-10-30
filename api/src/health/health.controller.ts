import { Controller, Get } from '@nestjs/common';
import { ApiOkResponse, ApiProperty, ApiTags } from '@nestjs/swagger';
import { ModelService } from '../model/model.service';

export class HealthDto {
  @ApiProperty({ example: 'ok' }) status!: 'ok';
  @ApiProperty({ example: 'ipl-chase-hgb-d81f5e586f86' }) modelVersion!: string;
}

@ApiTags('health')
@Controller('health')
export class HealthController {
  constructor(private readonly model: ModelService) {}

  @Get()
  @ApiOkResponse({ type: HealthDto })
  health(): HealthDto {
    // The app only starts once the model has loaded and passed its checks.
    return { status: 'ok', modelVersion: this.model.version };
  }
}
