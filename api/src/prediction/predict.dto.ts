import { ApiProperty } from '@nestjs/swagger';
import {
  IsInt, Max, Min, registerDecorator, ValidationArguments, ValidationOptions,
} from 'class-validator';
import { MAX_BALLS_BOWLED, MAX_TARGET, MAX_WICKETS_LOST, MESSAGES } from './contract';

/** score <= target - 1. Skipped when target itself is invalid; that error is reported instead. */
function IsBelowTarget(options?: ValidationOptions) {
  return (object: object, propertyName: string) => registerDecorator({
    name: 'isBelowTarget',
    target: object.constructor,
    propertyName,
    options,
    validator: {
      validate(value: unknown, args: ValidationArguments): boolean {
        const target = (args.object as { target?: unknown }).target;
        const targetIsValid = typeof target === 'number' && Number.isInteger(target)
          && target >= 1 && target <= MAX_TARGET;
        if (!targetIsValid) {
          return true;
        }
        return typeof value === 'number' && value <= target - 1;
      },
    },
  });
}

// class-validator runs a property's decorators from the bottom up and, with
// stopAtFirstError, reports only the first failure. @IsInt is therefore last:
// a wrong type gets the type message, and only integers are range-checked.
export class PredictRequestDto {
  @ApiProperty({ description: 'Runs the chasing team needs to win (first-innings score + 1)',
    minimum: 1, maximum: MAX_TARGET, example: 180 })
  @Max(MAX_TARGET, { message: MESSAGES.target.range })
  @Min(1, { message: MESSAGES.target.range })
  @IsInt({ message: MESSAGES.target.type })
  target!: number;

  @ApiProperty({ description: 'Runs scored so far in the chase (at most target - 1)', minimum: 0, example: 95 })
  @IsBelowTarget({ message: MESSAGES.score.range })
  @Min(0, { message: MESSAGES.score.range })
  @IsInt({ message: MESSAGES.score.type })
  score!: number;

  @ApiProperty({ description: 'Legal balls bowled so far (wides and no-balls do not count)',
    minimum: 0, maximum: MAX_BALLS_BOWLED, example: 66 })
  @Max(MAX_BALLS_BOWLED, { message: MESSAGES.ballsBowled.range })
  @Min(0, { message: MESSAGES.ballsBowled.range })
  @IsInt({ message: MESSAGES.ballsBowled.type })
  ballsBowled!: number;

  @ApiProperty({ description: 'Wickets lost so far', minimum: 0, maximum: MAX_WICKETS_LOST, example: 3 })
  @Max(MAX_WICKETS_LOST, { message: MESSAGES.wicketsLost.range })
  @Min(0, { message: MESSAGES.wicketsLost.range })
  @IsInt({ message: MESSAGES.wicketsLost.type })
  wicketsLost!: number;
}

export class FeaturesDto {
  @ApiProperty({ example: 85 }) runsLeft!: number;
  @ApiProperty({ example: 54 }) ballsLeft!: number;
  @ApiProperty({ example: 7 }) wicketsLeft!: number;
  @ApiProperty({ description: 'Current run rate (runs per over)', example: 8.636363636363637 }) crr!: number;
  @ApiProperty({ description: 'Required run rate (runs per over)', example: 9.444444444444445 }) rrr!: number;
}

export class PredictResponseDto {
  @ApiProperty({ description: 'Probability that the chasing team wins, between 0 and 1 (not rounded)',
    example: 0.6356436014175415 })
  winProbability!: number;

  @ApiProperty({ example: 'ipl-chase-hgb-d81f5e586f86' })
  modelVersion!: string;

  @ApiProperty({ type: FeaturesDto, description: 'Features the model received, derived from the request' })
  features!: FeaturesDto;
}

export class ValidationErrorDto {
  @ApiProperty({ example: 400 }) statusCode!: number;
  @ApiProperty({ example: ['target must be between 1 and 300', 'wicketsLost must be an integer'] })
  message!: string[];
  @ApiProperty({ example: 'Bad Request' }) error!: string;
}
