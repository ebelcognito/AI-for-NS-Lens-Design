/*
 * Copyright 2025 The Ray Optics Simulation authors and contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const readline = require('readline');
const { runSingleSimulation } = require('./simulation_core.js');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

let queue = Promise.resolve();

async function handleLine(rawLine) {
  const line = rawLine.trim();
  if (!line) {
    return;
  }

  if (line === 'QUIT') {
    process.exit(0);
  }

  const parseStart = Date.now();
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    console.log(JSON.stringify({
      error: 'invalid_json',
      details: error.message
    }));
    return;
  }

  const parseEnd = Date.now();

  try {
    const sceneJson = request.scene || request;
    const result = await runSingleSimulation(sceneJson);
    if (result.timing) {
      result.timing.json_parse_ms = parseEnd - parseStart;
    }

    const response = {
      result
    };

    if (Object.prototype.hasOwnProperty.call(request, 'episode')) {
      response.episode = request.episode;
    }
    if (Object.prototype.hasOwnProperty.call(request, 'step')) {
      response.step = request.step;
    }

    console.log(JSON.stringify(response));
  } catch (error) {
    const response = {
      error: 'simulation_failed',
      details: error.message
    };
    if (Object.prototype.hasOwnProperty.call(request, 'episode')) {
      response.episode = request.episode;
    }
    if (Object.prototype.hasOwnProperty.call(request, 'step')) {
      response.step = request.step;
    }
    console.log(JSON.stringify(response));
  }
}

rl.on('line', (line) => {
  queue = queue
    .then(() => handleLine(line))
    .catch((error) => {
      console.log(JSON.stringify({
        error: 'internal_error',
        details: error.message
      }));
    });
});

rl.on('close', () => {
  process.exit(0);
});

process.on('uncaughtException', (error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
});
