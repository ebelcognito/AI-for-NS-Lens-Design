// Interactive Ray Gradient & LM Refinement Demo Engine
// Antigravity AI - Lens Design

(function() {
  'use strict';

  // --- STATE ---
  const state = {
    // Optical parameters: [p0: Tilt, p1: Defocus, p2: Astig, p3: Coma, p4: Spherical]
    params: [0.15, -0.40, 0.25, -0.30, 0.50],
    numRays: 15,
    selectedParam: 1, // default selected is Defocus (p1)
    damp: 0.001,
    lr: 0.15,
    
    // Optimizer race state
    race: {
      running: false,
      step: 0,
      p_gd: [0.15, -0.40, 0.25, -0.30, 0.50],
      p_naive: [0.15, -0.40, 0.25, -0.30, 0.50],
      p_perray: [0.15, -0.40, 0.25, -0.30, 0.50],
      history_gd: [],
      history_naive: [],
      history_perray: [],
      timer: null
    }
  };

  const paramNames = [
    { name: "p0: Image Tilt / Shift", desc: "Constant shift across pupil (dr/dp0 = 1)" },
    { name: "p1: Mirror Curvature (Defocus)", desc: "Linear with pupil rho (dr/dp1 = rho)" },
    { name: "p2: Astigmatism (Freeform X^2-Y^2)", desc: "Quadratic with pupil (dr/dp2 = rho^2)" },
    { name: "p3: Coma (Freeform Asymmetry)", desc: "Cubic with pupil (dr/dp3 = rho^3)" },
    { name: "p4: Spherical Aberration (Asphere)", desc: "Quartic with pupil (dr/dp4 = rho^4)" }
  ];

  // --- OPTICAL PHYSICS & MATH FUNCTIONS ---

  function getPupilCoords(count) {
    if (count <= 1) return [0];
    const coords = [];
    for (let i = 0; i < count; i++) {
      coords.push(-1.0 + (2.0 * i) / (count - 1));
    }
    return coords;
  }

  // Transverse ray aberration error r_i(rho)
  function rayError(p, rho) {
    return p[0] + p[1]*rho + p[2]*(rho**2) + p[3]*(rho**3) + p[4]*(rho**4);
  }

  // Derivative d(r_i) / d(p_j)
  function rayDerivative(j, rho) {
    return Math.pow(rho, j);
  }

  // Compute Jacobians & RMS
  function evaluateSystem(p, numRays) {
    const rhos = getPupilCoords(numRays);
    const M = rhos.length;
    const N = 5;
    
    // Per-ray residuals
    const r = new Float64Array(M);
    let sumSq = 0;
    for (let i = 0; i < M; i++) {
      r[i] = rayError(p, rhos[i]);
      sumSq += r[i] * r[i];
    }
    const rms = Math.sqrt(sumSq / M);

    // Per-ray Jacobian J_ray: (M x N)
    const J_ray = [];
    for (let i = 0; i < M; i++) {
      const row = new Float64Array(N);
      for (let j = 0; j < N; j++) {
        row[j] = rayDerivative(j, rhos[i]);
      }
      J_ray.push(row);
    }

    // Scalar Jacobian J_scalar: (1 x N)
    // d(RMS)/d(p_j) = (1 / (M * RMS)) sum_i (r_i * J_ray[i, j])
    const J_scalar = new Float64Array(N);
    if (rms > 1e-12) {
      for (let j = 0; j < N; j++) {
        let dot = 0;
        for (let i = 0; i < M; i++) {
          dot += r[i] * J_ray[i][j];
        }
        J_scalar[j] = dot / (M * rms);
      }
    }

    // Gauss-Newton Hessian for Per-Ray LM: H_ray = (J_ray^T @ J_ray) / M
    const H_ray = createMatrix(N, N);
    for (let j = 0; j < N; j++) {
      for (let k = 0; k < N; k++) {
        let dot = 0;
        for (let i = 0; i < M; i++) {
          dot += J_ray[i][j] * J_ray[i][k];
        }
        H_ray[j][k] = dot / M;
      }
    }

    // Gauss-Newton Hessian for Naive LM: H_scalar = J_scalar^T @ J_scalar
    const H_scalar = createMatrix(N, N);
    for (let j = 0; j < N; j++) {
      for (let k = 0; k < N; k++) {
        H_scalar[j][k] = J_scalar[j] * J_scalar[k];
      }
    }

    return { rhos, r, rms, J_ray, J_scalar, H_ray, H_scalar, M, N };
  }

  function createMatrix(rows, cols) {
    const m = [];
    for (let i = 0; i < rows; i++) m.push(new Float64Array(cols));
    return m;
  }

  // Jacobi Eigenvalue Algorithm for real symmetric 5x5 matrix
  function eigenSymmetric(A, maxIter = 50) {
    const n = A.length;
    const V = createMatrix(n, n);
    for (let i = 0; i < n; i++) V[i][i] = 1.0;
    
    // Copy A
    const D = createMatrix(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) D[i][j] = A[i][j];
    }

    for (let iter = 0; iter < maxIter; iter++) {
      let maxOff = 0;
      let p = 0, q = 1;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(D[i][j]) > maxOff) {
            maxOff = Math.abs(D[i][j]);
            p = i; q = j;
          }
        }
      }
      if (maxOff < 1e-12) break;

      const diff = D[q][q] - D[p][p];
      let t;
      if (Math.abs(D[p][q]) < Math.abs(diff) * 1e-15) {
        t = D[p][q] / diff;
      } else {
        const phi = diff / (2.0 * D[p][q]);
        t = 1.0 / (Math.abs(phi) + Math.sqrt(phi * phi + 1.0));
        if (phi < 0) t = -t;
      }

      const c = 1.0 / Math.sqrt(t * t + 1.0);
      const s = t * c;
      const tau = s / (1.0 + c);

      const d_pq = D[p][q];
      D[p][q] = 0;
      D[p][p] -= t * d_pq;
      D[q][q] += t * d_pq;

      for (let j = 0; j < p; j++) {
        const d_jp = D[j][p], d_jq = D[j][q];
        D[j][p] = d_jp - s * (d_jq + tau * d_jp);
        D[j][q] = d_jq + s * (d_jp - tau * d_jq);
      }
      for (let j = p + 1; j < q; j++) {
        const d_pj = D[p][j], d_jq = D[j][q];
        D[p][j] = d_pj - s * (d_jq + tau * d_pj);
        D[j][q] = d_jq + s * (d_pj - tau * d_jq);
      }
      for (let j = q + 1; j < n; j++) {
        const d_pj = D[p][j], d_qj = D[q][j];
        D[p][j] = d_pj - s * (d_qj + tau * d_pj);
        D[q][j] = d_qj + s * (d_pj - tau * d_qj);
      }

      for (let j = 0; j < n; j++) {
        const v_jp = V[j][p], v_jq = V[j][q];
        V[j][p] = v_jp - s * (v_jq + tau * v_jp);
        V[j][q] = v_jq + s * (v_jp - tau * v_jq);
      }
    }

    const eigenvalues = [];
    for (let i = 0; i < n; i++) eigenvalues.push(Math.max(0, D[i][i]));
    eigenvalues.sort((a, b) => b - a);
    return eigenvalues;
  }

  // Gaussian elimination solve: (A + damp*I) x = b
  function solveLinear(A, b, damp = 0.001) {
    const n = b.length;
    const M = createMatrix(n, n + 1);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        M[i][j] = A[i][j] + (i === j ? damp : 0);
      }
      M[i][n] = b[i];
    }

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
      }
      const tmp = M[i]; M[i] = M[maxRow]; M[maxRow] = tmp;

      const pivot = M[i][i];
      if (Math.abs(pivot) < 1e-12) continue;
      for (let j = i; j <= n; j++) M[i][j] /= pivot;

      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = M[k][i];
          for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
        }
      }
    }

    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = M[i][n];
    return x;
  }

  // --- RENDERING ROUTINES ---

  const rayCanvas = document.getElementById('rayTraceCanvas');
  const rayCtx = rayCanvas.getContext('2d');

  const spotCanvas = document.getElementById('spotCanvas');
  const spotCtx = spotCanvas.getContext('2d');

  const spectrumCanvas = document.getElementById('spectrumCanvas');
  const specCtx = spectrumCanvas.getContext('2d');

  const convergenceCanvas = document.getElementById('convergenceCanvas');
  const convCtx = convergenceCanvas.getContext('2d');

  function renderRayTrace(evalResult) {
    const width = rayCanvas.width;
    const height = rayCanvas.height;
    rayCtx.clearRect(0, 0, width, height);

    // Draw Optical System Diagram
    // Layout: Entrance Pupil (left) -> Fold Mirror 1 -> Curved Grating Mirror 2 -> Focusing Mirror 3 -> Detector (right)
    const pupilX = 60;
    const m1X = 170, m1Y = 90;
    const m2X = 300, m2Y = 220;
    const m3X = 430, m3Y = 110;
    const detX = 540, detY = 200;

    // Background grid / axis
    rayCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    rayCtx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      rayCtx.beginPath(); rayCtx.moveTo(x, 0); rayCtx.lineTo(x, height); rayCtx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      rayCtx.beginPath(); rayCtx.moveTo(0, y); rayCtx.lineTo(width, y); rayCtx.stroke();
    }

    // Mirror 1
    drawMirror(rayCtx, m1X, m1Y, 70, -0.6, "M1: Fold / Collimator");
    // Mirror 2 (Grating)
    drawMirror(rayCtx, m2X, m2Y, 80, 0.5, "M2: Curved Grating", "#fbbf24");
    // Mirror 3 (Freeform)
    drawMirror(rayCtx, m3X, m3Y, 75, -0.4, "M3: Freeform Focus", "#a855f7");
    // Detector plane
    rayCtx.strokeStyle = '#38bdf8';
    rayCtx.lineWidth = 4;
    rayCtx.beginPath();
    rayCtx.moveTo(detX, detY - 60);
    rayCtx.lineTo(detX, detY + 60);
    rayCtx.stroke();
    rayCtx.fillStyle = '#38bdf8';
    rayCtx.font = '11px monospace';
    rayCtx.fillText("Detector", detX - 15, detY - 68);

    // Slit / Pupil
    rayCtx.strokeStyle = '#94a3b8';
    rayCtx.lineWidth = 3;
    rayCtx.beginPath();
    rayCtx.moveTo(pupilX, 110);
    rayCtx.lineTo(pupilX, 190);
    rayCtx.stroke();
    rayCtx.fillStyle = '#94a3b8';
    rayCtx.fillText("Slit Pupil", pupilX - 25, 95);

    // Trace rays
    const { rhos, r, M } = evalResult;
    for (let i = 0; i < M; i++) {
      const rho = rhos[i];
      const startY = 150 + rho * 35;
      
      // Calculate ray path hitting mirrors with freeform sag perturbation
      const p1HitY = m1Y + rho * 22;
      const p2HitY = m2Y - rho * 25 + state.params[1] * rho * 5;
      const p3HitY = m3Y + rho * 20 - state.params[4] * (rho**3) * 6;
      
      // Final landing on detector
      const hitDetY = detY + r[i] * 55; // scale aberration error

      // Ray color based on pupil position rho
      const isChief = Math.abs(rho) < 1e-4;
      let strokeColor = `hsla(${200 + rho * 45}, 90%, 65%, ${isChief ? 0.95 : 0.6})`;
      if (isChief) strokeColor = '#34d399'; // Green chief ray

      rayCtx.strokeStyle = strokeColor;
      rayCtx.lineWidth = isChief ? 2.2 : 1.3;
      rayCtx.beginPath();
      rayCtx.moveTo(pupilX, startY);
      rayCtx.lineTo(m1X, p1HitY);
      rayCtx.lineTo(m2X, p2HitY);
      rayCtx.lineTo(m3X, p3HitY);
      rayCtx.lineTo(detX, hitDetY);
      rayCtx.stroke();

      // Draw landing point
      rayCtx.fillStyle = isChief ? '#34d399' : '#38bdf8';
      rayCtx.beginPath();
      rayCtx.arc(detX, hitDetY, isChief ? 3.5 : 2.2, 0, Math.PI * 2);
      rayCtx.fill();
    }
  }

  function drawMirror(ctx, x, y, len, angle, label, color = "#94a3b8") {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    // slight curved mirror arc
    ctx.moveTo(0, -len/2);
    ctx.quadraticCurveTo(8, 0, 0, len/2);
    ctx.stroke();

    // Mirror hatching on back
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    for (let h = -len/2 + 5; h < len/2; h += 10) {
      ctx.beginPath();
      ctx.moveTo(2, h);
      ctx.lineTo(8, h - 4);
      ctx.stroke();
    }
    ctx.restore();

    ctx.fillStyle = color;
    ctx.font = '10px sans-serif';
    ctx.fillText(label, x - 30, y + len/2 + 15);
  }

  // Spot diagram and Per-Ray sensitivity arrow visualizer
  function renderSpotDiagram(evalResult) {
    const width = spotCanvas.width;
    const height = spotCanvas.height;
    spotCtx.clearRect(0, 0, width, height);

    const centerY = height / 2;
    const centerX = width / 2;

    // Draw detector scale and target focal line
    spotCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    spotCtx.lineWidth = 1;
    spotCtx.beginPath();
    spotCtx.moveTo(0, centerY); spotCtx.lineTo(width, centerY);
    spotCtx.moveTo(centerX, 0); spotCtx.lineTo(centerX, height);
    spotCtx.stroke();

    // Target focal circle (diffraction spot limit ~ 2 um)
    spotCtx.strokeStyle = 'rgba(52, 211, 153, 0.4)';
    spotCtx.setLineDash([4, 4]);
    spotCtx.beginPath();
    spotCtx.arc(centerX, centerY, 8, 0, Math.PI * 2);
    spotCtx.stroke();
    spotCtx.setLineDash([]);
    spotCtx.fillStyle = 'rgba(52, 211, 153, 0.8)';
    spotCtx.font = '10px monospace';
    spotCtx.fillText("Diffraction Limit", centerX + 12, centerY - 12);

    const { rhos, r, M } = evalResult;
    const selParam = state.selectedParam;
    const scaleY = 90; // visual pixel scale

    for (let i = 0; i < M; i++) {
      const rho = rhos[i];
      const err = r[i];
      const rayY = centerY + err * scaleY;
      const rayX = centerX + rho * 65; // spread pupil horizontally

      const isChief = Math.abs(rho) < 1e-4;
      spotCtx.fillStyle = isChief ? '#34d399' : '#38bdf8';
      spotCtx.beginPath();
      spotCtx.arc(rayX, rayY, isChief ? 4.5 : 3, 0, Math.PI * 2);
      spotCtx.fill();

      // PER-RAY SENSITIVITY ARROW: d(r_i) / d(p_selected)
      const deriv = rayDerivative(selParam, rho);
      const arrowLen = -deriv * 35; // sensitivity vector length

      if (Math.abs(deriv) > 1e-4) {
        spotCtx.strokeStyle = '#f43f5e';
        spotCtx.lineWidth = 1.8;
        spotCtx.beginPath();
        spotCtx.moveTo(rayX, rayY);
        spotCtx.lineTo(rayX, rayY + arrowLen);
        spotCtx.stroke();

        // Arrow head
        const head = arrowLen > 0 ? 4 : -4;
        spotCtx.fillStyle = '#f43f5e';
        spotCtx.beginPath();
        spotCtx.moveTo(rayX - 3, rayY + arrowLen - head);
        spotCtx.lineTo(rayX + 3, rayY + arrowLen - head);
        spotCtx.lineTo(rayX, rayY + arrowLen);
        spotCtx.fill();

        // Label sensitivity value on hover or endpoints
        if (i === 0 || i === M - 1 || isChief) {
          spotCtx.font = '9px monospace';
          spotCtx.fillStyle = '#fda4af';
          spotCtx.fillText(`dr/dp=${deriv.toFixed(2)}`, rayX - 20, rayY + arrowLen + (arrowLen > 0 ? 12 : -6));
        }
      }
    }
  }

  // Render Eigenvalue spectrum & Matrices
  function renderSpectrum(evalResult) {
    const width = spectrumCanvas.width;
    const height = spectrumCanvas.height;
    specCtx.clearRect(0, 0, width, height);

    const eigRay = eigenSymmetric(evalResult.H_ray);
    const eigScalar = eigenSymmetric(evalResult.H_scalar);

    const barWidth = 36;
    const gap = 16;
    const startX = 60;
    const baseY = height - 40;
    const maxVal = Math.max(...eigRay, ...eigScalar, 1.0);
    const maxBarH = height - 80;

    // Draw baseline
    specCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    specCtx.beginPath();
    specCtx.moveTo(20, baseY);
    specCtx.lineTo(width - 20, baseY);
    specCtx.stroke();

    specCtx.font = '11px sans-serif';
    specCtx.fillStyle = '#94a3b8';
    specCtx.fillText("Eigenvalue Spectrum of Gauss-Newton Curvature H = JᵀJ", 20, 22);

    for (let k = 0; k < 5; k++) {
      const x = startX + k * (barWidth * 2 + gap);

      // 1. Plain LM eigenvalue bar (RED)
      const valScalar = eigScalar[k] || 0;
      const hScalar = (valScalar / maxVal) * maxBarH;
      specCtx.fillStyle = '#f43f5e';
      specCtx.fillRect(x, baseY - hScalar, barWidth - 2, hScalar);
      if (valScalar < 1e-4) {
        specCtx.fillStyle = 'rgba(244, 63, 94, 0.6)';
        specCtx.fillText("0.0 (BLIND)", x - 4, baseY - 6);
      } else {
        specCtx.fillStyle = '#fecdd3';
        specCtx.fillText(valScalar.toFixed(2), x, baseY - hScalar - 6);
      }

      // 2. Per-Ray LM eigenvalue bar (GREEN / CYAN)
      const valRay = eigRay[k] || 0;
      const hRay = (valRay / maxVal) * maxBarH;
      specCtx.fillStyle = '#34d399';
      specCtx.fillRect(x + barWidth, baseY - hRay, barWidth - 2, hRay);
      specCtx.fillStyle = '#a7f3d0';
      specCtx.fillText(valRay.toFixed(2), x + barWidth, baseY - hRay - 6);

      // Axis label
      specCtx.fillStyle = '#64748b';
      specCtx.font = '10px monospace';
      specCtx.fillText(`λ${k+1}`, x + barWidth / 2 + 8, baseY + 18);
    }

    // Legend
    specCtx.fillStyle = '#f43f5e';
    specCtx.fillRect(width - 220, 15, 12, 12);
    specCtx.fillStyle = '#cbd5e1';
    specCtx.fillText("Plain LM (Rank 1)", width - 200, 25);

    specCtx.fillStyle = '#34d399';
    specCtx.fillRect(width - 220, 32, 12, 12);
    specCtx.fillStyle = '#cbd5e1';
    specCtx.fillText("Per-Ray LM (Full Rank 5)", width - 200, 42);

    // Update Rank / Condition status
    const rankScalar = eigScalar.filter(e => e > 1e-4).length;
    const rankRay = eigRay.filter(e => e > 1e-4).length;
    
    document.getElementById('rankScalarBadge').textContent = `Rank: ${rankScalar} / 5 (Degenerate)`;
    document.getElementById('rankRayBadge').textContent = `Rank: ${rankRay} / 5 (${rankRay === 5 ? 'Full Rank' : 'Deficient'})`;

    const condRay = eigRay[0] / Math.max(eigRay[4], 1e-12);
    document.getElementById('condRayPill').textContent = condRay > 1e5 ? 'Ill-conditioned' : condRay.toFixed(1);

    // Aliasing warning if M < 5
    const aliasBox = document.getElementById('aliasingAlert');
    if (evalResult.M < 5) {
      aliasBox.style.display = 'flex';
      aliasBox.innerHTML = `<strong>⚠️ SEVERE OPTICAL ALIASING!</strong> Number of rays (M = ${evalResult.M}) is less than parameters (N = 5). Hessian has ${5 - rankRay} blind null modes! The optimizer will blow up untraced pupil regions!`;
    } else {
      aliasBox.style.display = 'none';
    }

    // Render Matrix grids
    renderMatrixHTML('matrixScalar', evalResult.H_scalar);
    renderMatrixHTML('matrixRay', evalResult.H_ray);
  }

  function renderMatrixHTML(elementId, matrix) {
    const el = document.getElementById(elementId);
    el.innerHTML = '';
    const maxVal = Math.max(...matrix.map(row => Math.max(...row)), 0.001);

    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const cell = document.createElement('div');
        cell.className = 'matrix-cell';
        const val = matrix[r][c];
        const intensity = Math.min(Math.abs(val) / maxVal, 1.0);
        
        if (elementId === 'matrixScalar') {
          cell.style.backgroundColor = `rgba(244, 63, 94, ${0.15 + intensity * 0.7})`;
        } else {
          cell.style.backgroundColor = `rgba(52, 211, 153, ${0.15 + intensity * 0.7})`;
        }
        cell.textContent = Math.abs(val) < 1e-3 ? '0' : val.toFixed(2);
        cell.title = `H[${r},${c}] = ${val.toFixed(4)}`;
        el.appendChild(cell);
      }
    }
  }

  function renderConvergencePlot() {
    const width = convergenceCanvas.width;
    const height = convergenceCanvas.height;
    convCtx.clearRect(0, 0, width, height);

    const { history_gd, history_naive, history_perray } = state.race;
    if (history_gd.length === 0) {
      convCtx.fillStyle = '#64748b';
      convCtx.font = '12px sans-serif';
      convCtx.fillText("Click 'Auto-Run Race' or 'Step Optimization' to compare convergence curves.", 30, height / 2);
      return;
    }

    const maxSteps = Math.max(15, history_gd.length);
    const startX = 50, endX = width - 20;
    const startY = 20, endY = height - 30;

    // Axis
    convCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    convCtx.beginPath();
    convCtx.moveTo(startX, startY); convCtx.lineTo(startX, endY);
    convCtx.lineTo(endX, endY);
    convCtx.stroke();

    convCtx.fillStyle = '#94a3b8';
    convCtx.font = '10px monospace';
    convCtx.fillText("Iter 0", startX - 10, endY + 16);
    convCtx.fillText(`Iter ${history_gd.length - 1}`, endX - 35, endY + 16);
    convCtx.fillText("RMS Blur", startX - 45, startY + 10);

    const maxRMS = Math.max(history_gd[0], 0.6);

    function drawLine(hist, color) {
      convCtx.strokeStyle = color;
      convCtx.lineWidth = 2.2;
      convCtx.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const x = startX + (i / (maxSteps - 1)) * (endX - startX);
        const y = endY - (hist[i] / maxRMS) * (endY - startY);
        if (i === 0) convCtx.moveTo(x, y);
        else convCtx.lineTo(x, y);
      }
      convCtx.stroke();
    }

    drawLine(history_gd, '#f59e0b');       // Amber GD
    drawLine(history_naive, '#f43f5e');    // Red Plain LM
    drawLine(history_perray, '#34d399');   // Green Per-Ray LM

    // Legend
    convCtx.fillStyle = '#f59e0b'; convCtx.fillText("GD (Slow)", endX - 180, 25);
    convCtx.fillStyle = '#f43f5e'; convCtx.fillText("Plain LM (Stalls)", endX - 180, 42);
    convCtx.fillStyle = '#34d399'; convCtx.fillText("Per-Ray LM (Fast)", endX - 180, 59);
  }

  // --- REFRESH ALL ---
  function update() {
    const evalResult = evaluateSystem(state.params, state.numRays);

    document.getElementById('rmsVal').textContent = evalResult.rms.toFixed(5);
    document.getElementById('rayCountVal').textContent = state.numRays;
    document.getElementById('selParamDesc').textContent = paramNames[state.selectedParam].desc;

    // Sliders text
    for (let i = 0; i < 5; i++) {
      document.getElementById(`val_p${i}`).textContent = state.params[i].toFixed(2);
    }

    renderRayTrace(evalResult);
    renderSpotDiagram(evalResult);
    renderSpectrum(evalResult);
    renderConvergencePlot();
  }

  // --- OPTIMIZATION STEP ---
  function takeStep() {
    const { p_gd, p_naive, p_perray, history_gd, history_naive, history_perray } = state.race;
    const N = 5;
    const damp = state.damp;
    const lr = state.lr;
    const rays = state.numRays;

    // 1. Evaluate GD
    const eval_gd = evaluateSystem(p_gd, rays);
    history_gd.push(eval_gd.rms);
    const grad_gd = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let i = 0; i < eval_gd.M; i++) sum += eval_gd.r[i] * eval_gd.J_ray[i][j];
      grad_gd[j] = sum / eval_gd.M;
      p_gd[j] -= lr * grad_gd[j];
    }

    // 2. Evaluate Naive LM
    const eval_naive = evaluateSystem(p_naive, rays);
    history_naive.push(eval_naive.rms);
    const rhs_naive = new Float64Array(N);
    for (let j = 0; j < N; j++) rhs_naive[j] = eval_naive.J_scalar[j] * eval_naive.rms;
    const step_naive = solveLinear(eval_naive.H_scalar, rhs_naive, damp);
    for (let j = 0; j < N; j++) p_naive[j] -= step_naive[j];

    // 3. Evaluate Per-Ray LM
    const eval_perray = evaluateSystem(p_perray, rays);
    history_perray.push(eval_perray.rms);
    const rhs_perray = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let i = 0; i < eval_perray.M; i++) sum += eval_perray.r[i] * eval_perray.J_ray[i][j];
      rhs_perray[j] = sum / eval_perray.M;
    }
    const step_perray = solveLinear(eval_perray.H_ray, rhs_perray, damp);
    for (let j = 0; j < N; j++) p_perray[j] -= step_perray[j];

    state.race.step++;
    document.getElementById('raceStepCount').textContent = state.race.step;
    document.getElementById('raceRMS_gd').textContent = eval_gd.rms.toFixed(5);
    document.getElementById('raceRMS_naive').textContent = eval_naive.rms.toFixed(5);
    document.getElementById('raceRMS_perray').textContent = eval_perray.rms.toFixed(5);

    // Update active parameters to Per-Ray's state so canvas updates
    state.params = [...p_perray];
    update();
  }

  function resetRace() {
    if (state.race.timer) clearInterval(state.race.timer);
    state.race.running = false;
    state.race.step = 0;
    const init = [0.15, -0.40, 0.25, -0.30, 0.50];
    state.params = [...init];
    state.race.p_gd = [...init];
    state.race.p_naive = [...init];
    state.race.p_perray = [...init];
    state.race.history_gd = [];
    state.race.history_naive = [];
    state.race.history_perray = [];
    
    document.getElementById('btnAutoRace').textContent = "▶ Auto-Run Race";
    document.getElementById('raceStepCount').textContent = "0";
    document.getElementById('raceRMS_gd').textContent = "--";
    document.getElementById('raceRMS_naive').textContent = "--";
    document.getElementById('raceRMS_perray').textContent = "--";
    update();
  }

  function toggleAutoRace() {
    if (state.race.running) {
      clearInterval(state.race.timer);
      state.race.running = false;
      document.getElementById('btnAutoRace').textContent = "▶ Auto-Run Race";
    } else {
      if (state.race.step >= 20) resetRace();
      state.race.running = true;
      document.getElementById('btnAutoRace').textContent = "⏸ Pause Race";
      state.race.timer = setInterval(() => {
        if (state.race.step >= 18) {
          clearInterval(state.race.timer);
          state.race.running = false;
          document.getElementById('btnAutoRace').textContent = "↺ Re-run Race";
          return;
        }
        takeStep();
      }, 350);
    }
  }

  // --- EVENT LISTENERS ---
  function initListeners() {
    // Parameter sliders
    for (let i = 0; i < 5; i++) {
      const slider = document.getElementById(`slider_p${i}`);
      slider.addEventListener('input', (e) => {
        state.params[i] = parseFloat(e.target.value);
        state.selectedParam = i;
        document.getElementById(`selParamRadio_${i}`).checked = true;
        update();
      });

      const radio = document.getElementById(`selParamRadio_${i}`);
      radio.addEventListener('change', () => {
        state.selectedParam = i;
        update();
      });
    }

    // Number of rays slider
    const raySlider = document.getElementById('sliderRays');
    raySlider.addEventListener('input', (e) => {
      state.numRays = parseInt(e.target.value, 10);
      update();
    });

    // Preset buttons
    document.getElementById('btnPresetAberrated').addEventListener('click', () => {
      state.params = [0.15, -0.40, 0.25, -0.30, 0.50];
      for (let i = 0; i < 5; i++) document.getElementById(`slider_p${i}`).value = state.params[i];
      update();
    });

    document.getElementById('btnPresetFocused').addEventListener('click', () => {
      state.params = [0.0, 0.0, 0.0, 0.0, 0.0];
      for (let i = 0; i < 5; i++) document.getElementById(`slider_p${i}`).value = 0;
      update();
    });

    // Race buttons
    document.getElementById('btnStepRace').addEventListener('click', takeStep);
    document.getElementById('btnAutoRace').addEventListener('click', toggleAutoRace);
    document.getElementById('btnResetRace').addEventListener('click', resetRace);
  }

  // Initialize
  initListeners();
  update();

})();
