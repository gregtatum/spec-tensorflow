const tf = require('@tensorflow/tfjs')

const trainingDataSamples = 1000
const units = 256
const layers = 4
const trainingIterations = 15
const epochs = 100
const pixelsInChartStep = 5
const learningRate = 0.3;

const baseLayerConfig = {
  units,
  activation: 'relu',
  kernelInitializer: 'VarianceScaling',
}

const model = tf.sequential();
window.model = model

// Create the initial layer.
model.add(tf.layers.dense(Object.assign(baseLayerConfig, {
  name: 'hidden_0',
  inputShape: [2],
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
  optimizer: tf.train.sgd(learningRate),
  loss: 'meanSquaredError',
});

function formula (a, x) {
  return Math.sin((20 * a + Math.PI * 2) * x)
}

function generateTrainingData (samples) {
  const x = []
  const y = []
  for (let i = 0; i < samples; i++) {
    // Distribute the values so that there are more when the graph is denser
    const a = 1 - Math.pow(Math.random(), 5)
    const value = Math.random()
    x.push([a, value])
    y.push(formula(a, value))
  }
  return {
    x: tf.tensor(x, [samples, 2]),
    y: tf.tensor(y, [samples, 1])
  }
}

const { x, y } = generateTrainingData(trainingDataSamples);
const canvas = document.querySelector('canvas')
const ctx = canvas.getContext('2d')
const unitStep = pixelsInChartStep / canvas.width
const totalSteps = canvas.width / pixelsInChartStep
const scaleX = x => x * canvas.width
const scaleY = y => 0.3 * y * canvas.height + canvas.height / 2
const xsToGraph = Array(totalSteps).fill(0).map((_, i) => i / totalSteps)

async function drawCanvas () {
  let learnedYs
  const a = 0;
  const data = xsToGraph.map(x => ([a, x]))

  clearCanvas()
  drawFormula(a)

  console.time('Training the model')

  for (let i = 0; i < trainingIterations; i++) {
    console.log('Training', i)
    await model.fit(x, y, {
       batchSize: trainingDataSamples,
       epochs
    });

    learnedYs = model
      .predict(tf.tensor(data, [xsToGraph.length, data[0].length]))
      .dataSync()

    const hue = Math.floor(360 * i / trainingIterations)
    const alpha = (0.5 * i / trainingIterations) + 0.2
    drawModel(learnedYs, `hsla(${hue}, 50%, 50%, ${alpha})`)

    await wait(0)
  }

  console.timeEnd('Training the model')


  console.table(xsToGraph.map((x, i) => ({
    trainingYs: formula(a, x),
    learnedYs: learnedYs[i],
  })))
}

function clearCanvas () {
  // Clear the canvas
  ctx.fillStyle = '#eee'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
}

function drawFormula (a) {
  // Draw the base data
  ctx.strokeStyle = '#000'
  ctx.beginPath()
  for (let i = 0; i <= canvas.width; i += pixelsInChartStep) {
    const x1 = i / canvas.width
    const x2 = x1 + unitStep
    ctx.moveTo(scaleX(x1), scaleY(formula(a, x1)))
    ctx.lineTo(scaleX(x2), scaleY(formula(a, x2)))
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

function drawComplete (a) {
  const data = xsToGraph.map(x => ([a, x]))
  const learnedYs = model
    .predict(tf.tensor(data, [xsToGraph.length, 2]))
    .dataSync()

  clearCanvas()
  drawFormula(a)
  drawModel(learnedYs, `hsla(180, 50%, 50%)`)
}

drawCanvas(0);

document.querySelector('input').addEventListener('change', (e) => {
  drawComplete(Number(e.target.value || 0))
})

function wait(n) {
  return new Promise(resolve => setTimeout(resolve, n));
}
