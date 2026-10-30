import { join } from 'path';

/**
 * Default location of the model files: <repo>/ml/artifacts. This file sits at
 * api/src/config (or api/dist/config once built), three levels below the repo
 * root. Override with the MODEL_DIR environment variable.
 */
export const DEFAULT_MODEL_DIR = join(__dirname, '..', '..', '..', 'ml', 'artifacts');
