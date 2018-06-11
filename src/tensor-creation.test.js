const tf = require('@tensorflow/tfjs')
const { describe, it, expect, assert } = require('./test-framework')

describe('tensors', async function() {
  await it('can create a tensor', async function() {
    const tensor = tf.tensor([1, 2, 3, 10, 20, 30]);
    expect(await tensor.data()).toEqual([1, 2, 3, 10, 20, 30]);
    expect(tensor.shape).toEqual([6])
  });

  await it('can create a tensor with a shape', async function() {
    const tensor = tf.tensor([1, 2, 3, 10, 20, 30], [2, 3]);
    expect(await tensor.data()).toEqual([1, 2, 3, 10, 20, 30]);
    expect(tensor.shape).toEqual([2, 3])
  });

  await it('can create a tensor with an implicit shape', async function() {
    const tensor = tf.tensor([[1, 2, 3], [10, 20, 30]]);
    expect(await tensor.data()).toEqual([1, 2, 3, 10, 20, 30]);
    expect(tensor.shape).toEqual([2, 3])
  });

  await it('can create explicitly ranked tensors by method name', function() {
    expect(tf.tensor1d([0, 1, 2, 3]).shape).toEqual([4])
    expect(tf.tensor2d([0, 1, 2, 3], [2, 2]).shape).toEqual([2, 2])
    expect(tf.tensor3d([0, 1, 2, 3], [2, 2, 1]).shape).toEqual([2, 2, 1])
    expect(tf.tensor4d([0, 1, 2, 3], [2, 2, 1, 1]).shape).toEqual([2, 2, 1, 1])
  });

  await it('can clone a tensor', async function() {
    const tensor1 = tf.tensor1d([0, 1, 2, 3])
    const tensor2 = tensor1.clone()
    expect(await tensor1.data()).toEqual(await tensor2.data())
    assert(tensor1 !== tensor2)
  });

  await it('can create an identity matrix', async function() {
    const tensor = tf.eye(4, 4)
    expect(await tensor.data()).toEqual([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ])
  });

  await it('can fill a tensor with a value', async function() {
    const tensor = tf.fill([2, 2], 4)
    expect(await tensor.data()).toEqual([
      4, 4,
      4, 4
    ])
  });

  await it('can create a tensor via a linear function', async function() {
    const start = 0;
    const stop = 9;
    const number = 10;
    expect(await tf.linspace(0, 9, 10).data()).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ])
  })
})

describe('variables', async function() {
  await it('can assign to a variable', async function() {
    const variable = tf.variable(tf.tensor([0, 1, 2, 3]));
    expect(await variable.data()).toEqual([0, 1, 2, 3])
    variable.assign(tf.tensor([3, 2, 1, 0]))
    expect(await variable.data()).toEqual([3, 2, 1, 0])
  });

  await it('can assign the result of an operation', async function() {
    const variable = tf.variable(tf.tensor([0, 1, 2, 3]));
    variable.assign(variable.add(tf.tensor([1, 1, 1, 1])));
    expect(await variable.data()).toEqual([1, 2, 3, 4])
  });

});
