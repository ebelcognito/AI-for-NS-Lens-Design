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

const rayOptics = require('./rayOptics.js');

function hasCanvasModule() {
  try {
    require('canvas');
    return true;
  } catch (e) {
    return false;
  }
}

async function runSingleSimulation(sceneJson) {
  const t0_start = Date.now();

  const scene = new rayOptics.Scene();
  const t1_scene_create = Date.now();

  // NS-probe: the loader rejects unknown keys, so hand it a sanitized copy with our custom
  // fields stripped. They are re-applied to the live scene objects after load (see below).
  const cleanScene = Object.assign({}, sceneJson);
  delete cleanScene.recordInteractions;
  delete cleanScene.maxRayDepth;
  if (Array.isArray(sceneJson.objs)) {
    cleanScene.objs = sceneJson.objs.map((o) => {
      if (o && (o.reflectivity != null || o.reflectivityFront != null
                || o.reflectivityBack != null || o.elemCx != null)) {
        const c = Object.assign({}, o);
        delete c.reflectivity;
        delete c.reflectivityFront;
        delete c.reflectivityBack;
        delete c.elemCx;
        return c;
      }
      return o;
    });
  }

  await new Promise((resolve, reject) => {
    scene.loadJSON(JSON.stringify(cleanScene), function(needFullUpdate, completed) {
      if (completed) {
        resolve();
      } else if (scene.error) {
        reject(new Error(scene.error));
      }
    });
  });

  const t2_scene_load = Date.now();

  // --- NS-probe extension: opt-in per-ray interaction recording + forced per-surface reflectivity ---
  // These are no-ops unless the scene JSON requests them, so existing behaviour is unchanged.
  scene.recordInteractions = !!sceneJson.recordInteractions;
  scene.__raySeqRecords = [];
  if (sceneJson.maxRayDepth != null && isFinite(sceneJson.maxRayDepth)) {
    scene.maxRayDepth = sceneJson.maxRayDepth;
  }
  // Push designer-set `reflectivity` from the JSON onto the live scene objects (the deserializer
  // does not carry unknown properties). Honoured by BaseGlass.refract.
  if (Array.isArray(sceneJson.objs)) {
    scene.objs.forEach((obj, i) => {
      const src = sceneJson.objs[i];
      if (!src) return;
      let touched = false;
      if (src.reflectivity != null && isFinite(src.reflectivity)) {
        obj.reflectivity = src.reflectivity; touched = true;
      }
      if (src.reflectivityFront != null && isFinite(src.reflectivityFront)) {
        obj.reflectivityFront = src.reflectivityFront; touched = true;
      }
      if (src.reflectivityBack != null && isFinite(src.reflectivityBack)) {
        obj.reflectivityBack = src.reflectivityBack; touched = true;
      }
      if (src.elemCx != null && isFinite(src.elemCx)) {
        obj.elemCx = src.elemCx;
      }
      if (touched && obj.partialReflect == null) obj.partialReflect = true;
    });
  }
  // --- end NS-probe extension setup ---

  const cropBoxes = scene.objs.filter((obj) => obj.constructor.type === 'CropBox');
  const detectors = scene.objs.filter((obj) => obj.constructor.type === 'Detector');
  const hasCanvas = hasCanvasModule();

  if (cropBoxes.length > 0 && !hasCanvas) {
    throw new Error('Canvas module not found. To use image generation, install node-canvas.');
  }

  const result = {
    detectors: [],
    images: [],
    error: null,
    warning: null
  };

  const timing = {
    json_parse_ms: 0,
    scene_create_ms: t1_scene_create - t0_start,
    scene_load_ms: t2_scene_load - t1_scene_create,
    simulator_create_ms: 0,
    simulation_run_ms: 0,
    detector_extract_ms: 0,
    json_stringify_ms: 0,
    total_ms: 0
  };

  const errors = [];
  const warnings = [];

  async function simulateWithoutCanvas() {
    await new Promise((resolve) => {
      const t3_sim_create_start = Date.now();
      const simulator = new rayOptics.Simulator(scene);
      const t4_sim_create_end = Date.now();
      timing.simulator_create_ms += t4_sim_create_end - t3_sim_create_start;

      const t5_sim_run_start = Date.now();

      simulator.on('simulationComplete', () => {
        const t6_sim_run_end = Date.now();
        timing.simulation_run_ms += t6_sim_run_end - t5_sim_run_start;

        for (let i = 0; i < detectors.length; i++) {
          const detector = detectors[i];
          const detectorData = {
            name: detector.name || `Detector ${i + 1}`,
            power: detector.power,
            normal: detector.normal,
            shear: detector.shear
          };

          if (detector.irradMap && detector.binData) {
            detectorData.irradianceMap = detector.binData.map((value) => value / detector.binSize);
            detectorData.binPositions = detector.binData.map((_, j) => j * detector.binSize);
          }

          result.detectors.push(detectorData);
        }

        result.totalTruncation = simulator.totalTruncation;
        result.processedRayCount = simulator.processedRayCount;
        result.brightnessScale = simulator.brightnessScale;

        // NS-probe: per-run energy ledger (only present when recordInteractions was set).
        if (scene.recordInteractions && simulator.energyLedger) {
          result.energyLedger = simulator.energyLedger;
        }

        const t7_detector_extract = Date.now();
        timing.detector_extract_ms += t7_detector_extract - t6_sim_run_end;

        if (simulator.error) errors.push(`Simulator: ${simulator.error}`);
        if (simulator.warning) warnings.push(`Simulator: ${simulator.warning}`);

        resolve();
      });

      simulator.updateSimulation(false, false);
    });
  }

  async function simulateWithCanvas(cropBox, createCanvas, index) {
    await new Promise((resolve) => {
      const imageWidth = cropBox.width;
      const imageHeight = cropBox.width * (cropBox.p4.y - cropBox.p1.y) / (cropBox.p4.x - cropBox.p1.x);

      const canvasLight = createCanvas(imageWidth, imageHeight);
      const canvasBelowLight = createCanvas(imageWidth, imageHeight);
      const canvasAboveLight = createCanvas(imageWidth, imageHeight);
      const canvasGrid = createCanvas(imageWidth, imageHeight);
      const canvasVirtual = createCanvas(imageWidth, imageHeight);
      const canvasFinal = createCanvas(imageWidth, imageHeight);

      const ctxLight = canvasLight.getContext('2d');
      const ctxBelowLight = canvasBelowLight.getContext('2d');
      const ctxAboveLight = canvasAboveLight.getContext('2d');
      const ctxGrid = canvasGrid.getContext('2d');
      const ctxVirtual = canvasVirtual.getContext('2d');
      const ctxFinal = canvasFinal.getContext('2d');

      const originalScale = scene.scale;
      const originalOrigin = { ...scene.origin };
      const originalBackgroundColor = { ...scene.theme.background.color };

      scene.scale = cropBox.width / (cropBox.p4.x - cropBox.p1.x);
      scene.origin = { x: -cropBox.p1.x * scene.scale, y: -cropBox.p1.y * scene.scale };

      if (cropBox.transparent && scene.theme.background.color.r === 0 && scene.theme.background.color.g === 0 && scene.theme.background.color.b === 0) {
        scene.theme.background.color = { r: 0.01, g: 0.01, b: 0.01 };
      }

      const t3_sim_create_start = Date.now();
      const simulator = new rayOptics.Simulator(
        scene,
        ctxLight,
        ctxBelowLight,
        ctxAboveLight,
        ctxGrid,
        ctxVirtual,
        false
      );
      const t4_sim_create_end = Date.now();
      timing.simulator_create_ms += t4_sim_create_end - t3_sim_create_start;

      simulator.rayCountLimit = cropBox.rayCountLimit || Infinity;

      const t5_sim_run_start = Date.now();
      simulator.on('simulationComplete', () => {
        const t6_sim_run_end = Date.now();
        timing.simulation_run_ms += t6_sim_run_end - t5_sim_run_start;

        if (!cropBox.transparent) {
          ctxFinal.fillStyle = `rgb(${Math.round(scene.theme.background.color.r * 255)}, ${Math.round(scene.theme.background.color.g * 255)}, ${Math.round(scene.theme.background.color.b * 255)})`;
          ctxFinal.fillRect(0, 0, canvasFinal.width, canvasFinal.height);
        }

        ctxFinal.drawImage(canvasBelowLight, 0, 0);
        ctxFinal.drawImage(canvasGrid, 0, 0);
        ctxFinal.drawImage(canvasLight, 0, 0);
        ctxFinal.drawImage(canvasAboveLight, 0, 0);

        result.images.push({
          name: cropBox.name || `CropBox ${index + 1}`,
          dataUrl: canvasFinal.toDataURL('image/png')
        });

        scene.scale = originalScale;
        scene.origin = originalOrigin;
        scene.theme.background.color = originalBackgroundColor;

        result.totalTruncation = simulator.totalTruncation;
        result.processedRayCount = simulator.processedRayCount;
        result.brightnessScale = simulator.brightnessScale;

        // NS-probe: per-run energy ledger (only present when recordInteractions was set).
        if (scene.recordInteractions && simulator.energyLedger) {
          result.energyLedger = simulator.energyLedger;
        }

        if (simulator.error) errors.push(`Simulator: ${simulator.error}`);
        if (simulator.warning) warnings.push(`Simulator: ${simulator.warning}`);

        resolve();
      });

      simulator.updateSimulation(false, false);
    });
  }

  if (cropBoxes.length === 0) {
    await simulateWithoutCanvas();
  } else {
    const { createCanvas } = require('canvas');
    for (let i = 0; i < cropBoxes.length; i++) {
      await simulateWithCanvas(cropBoxes[i], createCanvas, i);
    }
    if (detectors.length > 0) {
      await simulateWithoutCanvas();
    }
  }

  if (scene.error) errors.push(`Scene: ${scene.error}`);
  if (scene.warning) warnings.push(`Scene: ${scene.warning}`);

  scene.objs.forEach((obj, i) => {
    if (obj.getError()) {
      errors.push(`objs[${i}] ${obj.constructor.type}: ${obj.getError()}`);
    }
    if (obj.getWarning()) {
      warnings.push(`objs[${i}] ${obj.constructor.type}: ${obj.getWarning()}`);
    }
  });

  result.error = errors.length > 0 ? errors.join('\n') : null;
  result.warning = warnings.length > 0 ? warnings.join('\n') : null;

  // NS-probe: expose recorded per-ray interaction sequences (empty unless recordInteractions was set).
  if (scene.recordInteractions) {
    result.raySequences = scene.__raySeqRecords || [];
  }

  const t8_before_stringify = Date.now();
  const resultJson = JSON.stringify(result);
  const t9_after_stringify = Date.now();
  timing.json_stringify_ms = t9_after_stringify - t8_before_stringify;
  timing.total_ms = t9_after_stringify - t0_start;

  // Consume the serialized output once so timing includes real stringify cost.
  if (!resultJson) {
    throw new Error('Failed to serialize simulation result');
  }

  result.timing = timing;
  return result;
}

module.exports = {
  runSingleSimulation
};
