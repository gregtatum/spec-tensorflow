const tf = require('@tensorflow/tfjs')


const trainingDataSamples = 300
const units = 128
const layers = 2
const trainingIterations = 25
const epochs = 100
const pixelsInChartStep = 5

const baseLayerConfig = {
  units,
  activation: 'relu',
  kernelInitializer: 'randomUniform',
  biasInitializer: 'randomUniform',
}

const model = tf.sequential();
window.model = model

// Create the initial layer.
model.add(tf.layers.dense(Object.assign(baseLayerConfig, {
  name: 'hidden_0',
  inputShape: [1],
})));

// Add as many layers as we want.
for (let i = 0; i < layers - 1; i++) {
  model.add(tf.layers.dense(Object.assign(baseLayerConfig, {
    name: `hidden_${i + 1}`,
  })));
}

// Try to guess a single scalar value out.
model.add(tf.layers.dense({
  units: 1
}));

// Compile the model so it can be used.
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError',
});

function formula (x) {
  x -= 0.5
  x *= 3
  const epx = Math.pow(Math.E, x)
  const enx = Math.pow(Math.E, -x)
  return (epx - enx) / (epx + enx)
}

function generateTrainingData (samples) {
  const x = []
  const y = []
  for (let i = 0; i < samples; i++) {
    const value = Math.random()
    x.push(value)
    y.push(formula(value))
  }
  return {
    x: tf.tensor(x, [samples, 1]),
    y: tf.tensor(y, [samples, 1])
  }
}

const { x, y } = generateTrainingData(trainingDataSamples);
const canvas = document.querySelector('canvas')
const ctx = canvas.getContext('2d')
const unitStep = pixelsInChartStep / canvas.width
const totalSteps = canvas.width / pixelsInChartStep
const scaleX = x => x * canvas.width
const scaleY = y => 0.5 * y * canvas.height + canvas.height / 2
const xsToGraph = Array(totalSteps).fill(0).map((_, i) => i / totalSteps)

async function drawCanvas () {
  let learnedYs
  clearCanvas()
  drawFormula()

  console.time('Training the model')

  for (let i = 0; i < trainingIterations; i++) {
    console.log('Training', i)
    await model.fit(x, y, {
       batchSize: trainingDataSamples,
       epochs
    });

    learnedYs = model
      .predict(tf.tensor(xsToGraph, [xsToGraph.length, 1]))
      .dataSync()

    const hue = Math.floor(360 * i / trainingIterations)
    const alpha = (0.5 * i / trainingIterations) + 0.2
    drawModel(learnedYs, `hsla(${hue}, 50%, 50%, ${alpha})`)

    await wait(0)
  }

  console.timeEnd('Training the model')


  console.table(xsToGraph.map((x, i) => ({
    trainingYs: formula(x),
    learnedYs: learnedYs[i],
  })))
}

function clearCanvas () {
  // Clear the canvas
  ctx.fillStyle = '#eee'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
}

function drawFormula () {
  // Draw the base data
  ctx.strokeStyle = '#000'
  ctx.beginPath()
  for (let i = 0; i <= canvas.width; i += pixelsInChartStep) {
    const x1 = i / canvas.width
    const x2 = x1 + unitStep
    ctx.moveTo(scaleX(x1), scaleY(formula(x1)))
    ctx.lineTo(scaleX(x2), scaleY(formula(x2)))
  }
  ctx.stroke()
}

function drawModel (ys, color) {
  // Draw the model data.
  ctx.strokeStyle = color
  ctx.beginPath()
  const step = canvas.width / (ys.length - 1)
  const unitX = step / canvas.width
  for (let i = 0; i < ys.length - 1; i++) {
    const x1 = unitX * i
    const x2 = x1 + 1 / (ys.length - 1)
    const y1 = ys[i]
    const y2 = ys[i + 1]
    ctx.moveTo(scaleX(x1), scaleY(y1))
    ctx.lineTo(scaleX(x2), scaleY(y2))
  }
  ctx.stroke()
}

drawCanvas(0);

;[...document.querySelectorAll('input')].forEach(
  input => input.addEventListener('change', drawCanvas)
)

function wait(n) {
  return new Promise(resolve => setTimeout(resolve, n));
}
