let tests = []
async function describe (name, fn) {
  tests.push({ name, fn })
}

requestAnimationFrame(async () => {
  for (const {name, fn} of tests) {
    console.log(name + '\n');
    await fn()
  }
});

async function it (name, fn) {
  try {
    await fn()
    console.log('%c ✔ ' + name, 'color: #0a0;')
  } catch (error) {
    console.log('%c ｘ ' + name, 'color: #a00;')
    console.error(error)
  }
}

function assert (isTrue, a, b) {
  if (!isTrue) {
    if (arguments.length === 3) {
      console.log('a:', a)
      console.log('b:', b)
      throw new Error(`"${JSON.stringify(a)}" is not equal to "${JSON.stringify(b)}"`)
    }
    throw new Error(`Assertion failed`)
  }
}

function expect (a) {
  return {
    toBe: b => assert(a === b, a, b),
    toEqual: b => {
      if (ArrayBuffer.isView(a)) a = [...a]
      if (ArrayBuffer.isView(b)) b = [...b]
      assert(JSON.stringify(a) === JSON.stringify(b), a, b)
    }
  }
}

module.exports = { it, assert, expect, describe }
