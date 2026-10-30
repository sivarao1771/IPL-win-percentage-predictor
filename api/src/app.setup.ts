import { INestApplication, ValidationPipe } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

/** Shared by main.ts and the e2e tests, so tests run the same pipes and routes as production. */
export function configureApp(app: INestApplication): void {
  app.setGlobalPrefix('api');
  app.useGlobalPipes(new ValidationPipe({
    whitelist: true,
    forbidNonWhitelisted: true, // reject unknown fields such as a misspelt "wicketLost"
    stopAtFirstError: true, // one message per field: type first, then range
    transform: true, // no implicit conversion: "5" (a string) is rejected, not coerced
  }));

  const origins = process.env.CORS_ORIGIN?.split(',').map((o) => o.trim()).filter(Boolean);
  if (origins?.length) {
    app.enableCors({ origin: origins });
  }

  const config = new DocumentBuilder()
    .setTitle('IPL Win Predictor API')
    .setDescription('Win probability for the chasing team in an IPL second innings. '
      + 'Model: gradient-boosted trees exported to ONNX; see ml/reports/results.md.')
    .setVersion('1.0')
    .build();
  SwaggerModule.setup('docs', app, SwaggerModule.createDocument(app, config));
}
