/** The parts of ml/artifacts/model_meta.json that the API relies on. */
export interface ContractRule {
  min: number;
  max: number | string;
  integer: boolean;
  message: string;
}

export interface ModelMeta {
  model_version: string;
  model_sha256: string;
  input: { name: string; dtype: string; feature_order: string[] };
  output: { name: string; dtype: string; win_probability_column: number };
  input_contract: Record<string, ContractRule>;
}
