// Background renderer worker — owns the OffscreenCanvas & runs the WebGL loop
// off the main thread. The main thread only sends control messages.

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';

// ── State (populated by messages from main) ────────────────────────────────
let canvas = null;
let renderer = null;
let scene = null;
let camera = null;
let mesh = null;
let uniforms = null;
let clock = null;
let running = true;
let timeScale = 4.0;
let rafId = null;

// ── Shaders ────────────────────────────────────────────────────────────────
const vertexShader = /* glsl */`
  void main() {
    gl_Position = vec4(position, 1.0);
  }
`;

const fragmentShader = /* glsl */`
  precision mediump float;
  uniform float uTime;
  uniform vec2  uResolution;
  uniform float uWarp;
  uniform float uBlob;
  uniform float uIntensity;
  uniform float uFlowRot;
  uniform float uColBreath;
  uniform float uHDR;
  uniform vec3  uC1, uC2, uC3, uC4, uC5;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),u.x),
               mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y);
  }
  float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    mat2 r = mat2(0.8,0.6,-0.6,0.8);
    s += a*noise(p); p = r*p*2.1; a *= 0.5;
    s += a*noise(p); p = r*p*2.1; a *= 0.5;
    s += a*noise(p);
    return s;
  }

  mat2 flowRotation(float t) {
    float ang = t * 0.010 * uFlowRot;
    return mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
  }

  float breath(int slot, float t) {
    float phase, rate;
    if      (slot == 0) { phase = 0.0; rate = 0.055; }
    else if (slot == 1) { phase = 1.3; rate = 0.063; }
    else if (slot == 2) { phase = 2.7; rate = 0.071; }
    else if (slot == 3) { phase = 4.1; rate = 0.079; }
    else                { phase = 5.5; rate = 0.087; }
    float b = 0.5 + 0.5 * sin(t * rate + phase);
    return mix(1.0, mix(0.35, 1.0, b), uColBreath);
  }

  // When uHDR=1, boost colour saturation and let values push above 1.0.
  // With display-p3 output colour space, the compositor maps these into the
  // wider P3 gamut on supported displays (Retina, most modern OLED phones).
  vec3 hdrBoost(vec3 c) {
    // Push each colour further from the ivory reference point
    vec3 ref = vec3(0.964, 0.952, 0.934);
    return ref + (c - ref) * mix(1.0, 1.55, uHDR);
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    vec2 st = uv;
    float t = uTime;

    vec2 warp = vec2(fbm(st * uBlob + t * 0.15), fbm(st * uBlob + 5.2 + t * 0.13));
    vec2 p = uv + (warp - 0.5) * uWarp * 0.08;

    vec3 ivory = vec3(0.964, 0.952, 0.934);

    mat2 rot = flowRotation(t);
    vec2 sampleP = rot * (st - 0.5) * uBlob * 0.7 + 0.5 + vec2(t * 0.08, t * 0.06);
    float nx = smoothstep(0.25, 0.75, fbm(sampleP));
    float ny = smoothstep(0.25, 0.75, fbm(sampleP + vec2(7.3, 4.1)));

    vec3 tint = vec3(0.0);
    tint += (hdrBoost(uC1) - ivory) * (1.0 - length(vec2(nx, ny) - vec2(0.15, 0.85))) * breath(0, t);
    tint += (hdrBoost(uC2) - ivory) * (1.0 - length(vec2(nx, ny) - vec2(0.85, 0.85))) * breath(1, t);
    tint += (hdrBoost(uC3) - ivory) * (1.0 - length(vec2(nx, ny) - vec2(0.50, 0.50))) * breath(2, t);
    tint += (hdrBoost(uC4) - ivory) * (1.0 - length(vec2(nx, ny) - vec2(0.15, 0.15))) * breath(3, t);
    tint += (hdrBoost(uC5) - ivory) * (1.0 - length(vec2(nx, ny) - vec2(0.85, 0.15))) * breath(4, t);

    // Only clamp when NOT in HDR mode — HDR path lets values exceed 1.0
    // and relies on the display-p3 colour-space canvas to preserve them.
    vec3 col = ivory + tint * uIntensity * 0.9;
    col = mix(clamp(col, 0.0, 1.0), col, uHDR);
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ── Setup: called once when main transfers the OffscreenCanvas ─────────────
function init({ canvas: transferredCanvas, width, height }) {
  canvas = transferredCanvas;
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    powerPreference: 'low-power',
    alpha: false,
    preserveDrawingBuffer: false,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  // Display-P3 output — on P3-capable displays this gives access to a wider
  // colour gamut. On sRGB displays it falls back gracefully.
  if (THREE.DisplayP3ColorSpace) {
    renderer.outputColorSpace = THREE.DisplayP3ColorSpace;
  }

  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  uniforms = {
    uTime:       { value: 0.0 },
    uResolution: { value: new THREE.Vector2(width, height) },
    uWarp:       { value: 1.5 },
    uBlob:       { value: 2.0 },
    uIntensity:  { value: 0.50 },
    uFlowRot:    { value: 1.0 },
    uColBreath:  { value: 1.0 },
    uHDR:        { value: 0.0 },
    uC1: { value: new THREE.Vector3(0.943, 0.696, 0.734) },
    uC2: { value: new THREE.Vector3(0.715, 0.856, 0.753) },
    uC3: { value: new THREE.Vector3(0.962, 0.829, 0.658) },
    uC4: { value: new THREE.Vector3(0.791, 0.734, 0.924) },
    uC5: { value: new THREE.Vector3(0.677, 0.848, 0.867) },
  };

  mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader })
  );
  scene.add(mesh);

  clock = new THREE.Clock();

  // Kick off the render loop
  tick();
}

function tick() {
  rafId = requestAnimationFrame(tick);
  if (!running) return;
  uniforms.uTime.value = clock.getElapsedTime() * timeScale;
  renderer.render(scene, camera);
}

// ── Message router ─────────────────────────────────────────────────────────
self.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      init(msg);
      break;
    case 'resize':
      if (!renderer) return;
      renderer.setSize(msg.width, msg.height, false);
      uniforms.uResolution.value.set(msg.width, msg.height);
      break;
    case 'uniform':
      if (!uniforms) return;
      if (msg.name === 'timeScale') {
        timeScale = msg.value;
      } else if (uniforms[msg.name]) {
        uniforms[msg.name].value = msg.value;
      }
      break;
    case 'visibility':
      running = msg.visible;
      if (running && clock) clock.getDelta();  // discard hidden-time delta
      break;
  }
};
