const tf = require('@tensorflow/tfjs')
const { describe, it, expect, assert } = require('./test-framework')

describe('simple operations', async function() {
  await it('can add together two tensors', async function() {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 5, 6]);
    const c = a.add(b);
    expect(await c.data()).toEqual([5, 7, 9]);
  });

  await it('can "broadcast" two tensors of different shapes', async function() {
    const a = tf.tensor([[1, 1, 1]]);
    const b = tf.tensor([[2, 2, 2], [3, 3, 3]]);
    const c = a.add(b);
    expect(await c.data()).toEqual([3, 3, 3, 4, 4, 4]);
  });

  await it('will not "broadcast" when using addStrict', async function() {
    const a = tf.tensor([[1, 1, 1]]);
    const b = tf.tensor([[2, 2, 2], [3, 3, 3]]);
    let didThrow = false
    try {
      a.addStrict(b);
    } catch (e) {
      didThrow = true;
    }
    expect(didThrow).toBe(true)
  });

  await it('can subtract two tensors', async function() {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 5, 6]);
    const c = a.sub(b);
    expect(await c.data()).toEqual([-3, -3, -3]);
  });

  await it('can multiply two tensors', async function() {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 5, 6]);
    const c = a.mul(b);
    expect(await c.data()).toEqual([4, 10, 18]);
  });

  await it('can divide two tensors', async function() {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 5, 6]);
    const c = a.div(b);
    const result = await c.data();
    // Take care for precision issues
    assert(Math.abs(result[0] - 0.25) < 0.00001)
    assert(Math.abs(result[1] - 0.4) < 0.00001)
    assert(Math.abs(result[2] - 0.5) < 0.00001)
  });

  await it('can find the maximum value of two tensors', async function() {
    const a = tf.tensor([1, 4, 5]);
    const b = tf.tensor([2, 3, 6]);
    const c = a.maximum(b);
    const result = await c.data();
    expect(await c.data()).toEqual([2, 4, 6]);
  });

  await it('can find the minimum value of two tensors', async function() {
    const a = tf.tensor([1, 4, 5]);
    const b = tf.tensor([2, 3, 6]);
    const c = a.minimum(b);
    const result = await c.data();
    expect(await c.data()).toEqual([1, 3, 5]);
  });

  await it('can find the modulo value of two tensors', async function() {
    const a = tf.tensor([5, 6, 7]);
    const b = tf.tensor([4, 4, 4]);
    const c = a.mod(b);
    const result = await c.data();
    expect(await c.data()).toEqual([1, 2, 3]);
  });
});

describe('matrix operations', async function() {
  await it('can compute the dot product', async function() {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 5, 6]);
    const c = a.dot(b);
    expect(await c.data()).toEqual([32]);
  })

  await it('can multiply two matrices together', async function() {
    const a = tf.tensor([
      2, 0, 0, 0,
      0, 2, 0, 0,
      0, 0, 2, 0,
      0, 0, 0, 2
    ], [4, 4]);
    const b = tf.tensor([
      2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 4, 0,
      7, 7, 7, 5
    ], [4, 4]);
    const c = a.matMul(b);
    expect(await c.data()).toEqual([
      4, 0, 0, 0,
      0, 6, 0, 0,
      0, 0, 8, 0,
      14, 14, 14, 10
    ]);
  })
});
