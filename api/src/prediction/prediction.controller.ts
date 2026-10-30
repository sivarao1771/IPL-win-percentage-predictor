import { Body, Controller, HttpCode, Post } from '@nestjs/common';
import { ApiBadRequestResponse, ApiOkResponse, ApiOperation, ApiTags } from '@nestjs/swagger';
import { PredictRequestDto, PredictResponseDto, ValidationErrorDto } from './predict.dto';
import { PredictionService } from './prediction.service';

@ApiTags('prediction')
@Controller('predict')
export class PredictionController {
  constructor(private readonly predictions: PredictionService) {}

  @Post()
  @HttpCode(200)
  @ApiOperation({ summary: 'Win probability for the chasing team in an IPL second innings' })
  @ApiOkResponse({ type: PredictResponseDto })
  @ApiBadRequestResponse({ type: ValidationErrorDto, description: 'Invalid or out-of-range input' })
  predict(@Body() body: PredictRequestDto): Promise<PredictResponseDto> {
    return this.predictions.predict(body);
  }
}
