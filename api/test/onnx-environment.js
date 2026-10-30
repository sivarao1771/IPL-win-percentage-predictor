// Jest environment for tests that run onnxruntime-node.
//
// Jest runs each test file in a separate JavaScript realm (a vm context) with its
// own Float32Array, but onnxruntime-node's native code creates its output arrays
// in Node's main realm. onnxruntime then checks `data instanceof Float32Array`
// against the test realm's constructor and fails. Sharing the main realm's
// typed-array constructors makes the check pass. Only tests need this; the
// built app runs in a single realm.
const { TestEnvironment } = require('jest-environment-node');

const SHARED = [
  'ArrayBuffer', 'Float32Array', 'Float64Array', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray',
  'Int16Array', 'Uint16Array', 'Int32Array', 'Uint32Array', 'BigInt64Array', 'BigUint64Array',
];

class OnnxEnvironment extends TestEnvironment {
  constructor(config, context) {
    super(config, context);
    for (const name of SHARED) {
      this.global[name] = globalThis[name];
    }
  }
}

module.exports = OnnxEnvironment;
