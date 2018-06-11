const tf = require('@tensorflow/tfjs')
const { describe, it, expect, assert } = require('./test-framework')

describe('models', function() {
  it('can create a model', function() {
    const model = tf.sequential();
    m=model

    model.add(tf.layers.dense({
      // 3 input values
      inputShape: [3],
      // 32 units in the shape
      units: 32,
      activation: 'sigmoid',
    }));

    // Define the final output
    model.add(tf.layers.dense({
      units: 4
    }));

    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError',
    });

    function formula (a, b, c) {
      return a * Math.pow(b, 3) + c
    }

    function generateTestData (count) {
      const x = []
      const y = []
      for (let i = 0; i < count; i++) {
        const a = Math.random()
        const b = Math.random()
        const c = Math.random()
        x.push(tf.tensor([a, b, c]))
        y.push(tf.tensor([formula(a, b, c)])
      }
      return { x, y }
    }

    const count = 200
    const { x, y } = generateTestData(count);

    model.fit(x, y, {
      batchSize: count,
      epochs: 100,
    });

    // Inspect the inferred shape of the model's output, which equals
    // `[null, 4]`. The 1st dimension is the undetermined batch dimension; the
    // 2nd is the output size of the model's last layer.
    console.log(JSON.stringify(model.outputs[0].shape));
  });

  // it('can create a neural network', function() {
  //   const inputs = tf.input({ shape: [5] });
  //   const denseLayer = tf.layers.dense({ units: 1 });
  //   const activationLayer = tf.layers.activation({ activation: 'sigmoid' });
  //
  //   // Apply the inputs to the layer
  //   const denseOutput = denseLayer.apply(inputs);
  //   const activationOutput = activationLayer.apply(denseOutput);
  //
  //   const model = tf.model({
  //     inputs,
  //     outputs: [denseOutput, activationOutput]
  //   });
  //
  //   model.predict(tf.randomNormal([6, 5]));
  //   denseOut.print();
  //   activationOut.print();
  // });
});
