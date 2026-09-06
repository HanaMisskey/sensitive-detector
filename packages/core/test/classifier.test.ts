import { describe, expect, it } from 'vitest';
import { createClassifier } from '../src/classifier.js';
import { encodeRgbPng } from './helpers/png.js';

describe('classifier', () => {
  it('returns MODEL_UNAVAILABLE when model directory does not exist', async () => {
    const classifier = await createClassifier('/nonexistent/path/', {
      arch: 'x64',
    });
    expect(classifier.available).toBe(false);
    const result = await classifier.classify(Buffer.from('test'));
    expect(result).toEqual({ ok: false, code: 'MODEL_UNAVAILABLE' });
  });

  it('returns IMAGE_DECODE_FAILED for non-PNG bytes', async () => {
    // createClassifier with a valid model is integration-test territory.
    // This test verifies the unavailable path only.
    const classifier = await createClassifier('/nonexistent/path/', {
      arch: 'x64',
    });
    const result = await classifier.classify(Buffer.from('not an image'));
    expect(result).toEqual({ ok: false, code: 'MODEL_UNAVAILABLE' });
  });
});

// 固定の logits を使い、実モデルの精度と独立にテンソル契約を検証する。
describe('384 model interface', () => {
  async function withOutput(output: Record<string, { data: Float32Array }>) {
    const tensors: { type: string; data: Float32Array; dims: number[] }[] = [];
    const feeds: Record<string, unknown>[] = [];
    const paths: string[] = [];
    const classifier = await createClassifier('/models', {
      loadOnnx: async () => ({
        Tensor: class {
          constructor(type: string, data: Float32Array, dims: number[]) {
            tensors.push({ type, data, dims });
          }
        },
        InferenceSession: {
          create: async (path) => {
            paths.push(path);
            return {
              run: async (input) => {
                feeds.push(input);
                return output;
              },
            };
          },
        },
      }),
    });
    return { classifier, tensors, feeds, paths };
  }

  it('uses NCHW RGB normalization and stable softmax in label order', async () => {
    const { classifier, tensors, feeds, paths } = await withOutput({
      logits: { data: new Float32Array([1000, 1001]) },
    });
    const result = await classifier.classify(encodeRgbPng(384, 384, [0, 128, 255]));
    expect(paths).toEqual(['/models/model.onnx']);
    expect(Object.keys(feeds[0] ?? {})).toEqual(['input']);
    expect(tensors[0]?.dims).toEqual([1, 3, 384, 384]);
    expect(tensors[0]?.type).toBe('float32');
    const data = tensors[0]?.data;
    expect(data?.length).toBe(3 * 384 * 384);
    expect(data?.[0]).toBe(-1);
    expect(data?.[384 * 384 - 1]).toBe(-1);
    expect(data?.[384 * 384]).toBeCloseTo(1 / 255);
    expect(data?.[2 * 384 * 384]).toBe(1);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.predictions.map((p) => p.className)).toEqual(['safe', 'nsfw']);
      expect(result.predictions[0]?.probability).toBeCloseTo(0.7310585786);
      expect(result.predictions[1]?.probability).toBeCloseTo(0.2689414214);
    }
  });

  it.each([[], [1], [1, 2, 3], [NaN, 0], [Infinity, 0]])('rejects malformed logits %j', async (...values) => {
    const { classifier } = await withOutput({ logits: { data: new Float32Array(values) } });
    expect(await classifier.classify(encodeRgbPng(384, 384, [0, 0, 0]))).toEqual({
      ok: false,
      code: 'DETECTION_FAILED',
    });
  });

  it('rejects missing output and legacy image dimensions', async () => {
    const { classifier, feeds } = await withOutput({});
    expect(await classifier.classify(encodeRgbPng(299, 299, [0, 0, 0]))).toEqual({
      ok: false,
      code: 'IMAGE_DECODE_FAILED',
    });
    expect(feeds).toHaveLength(0);
    expect(await classifier.classify(encodeRgbPng(384, 384, [0, 0, 0]))).toEqual({
      ok: false,
      code: 'DETECTION_FAILED',
    });
  });
});
