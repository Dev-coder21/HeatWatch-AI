// GLSL for the planet. Geographic convention everywhere in the scene:
// (lat, lon) -> (cos lat * sin lon, sin lat, cos lat * cos lon).

const NOISE = /* glsl */ `
// 3D simplex noise (Ashima Arts / Stefan Gustavson, MIT licence).
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
float fbm(vec3 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 6; i++) {
    v += a * snoise(p);
    p = p * 2.03 + vec3(1.7, 9.2, 3.1);
    a *= 0.5;
  }
  return v * 0.5 + 0.5;
}
`;

export const surfaceVertex = /* glsl */ `
varying vec3 vObj;
varying vec3 vNormalView;
varying vec3 vViewPos;
void main() {
  vObj = position;
  vNormalView = normalize(normalMatrix * normal);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewPos = mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

export const earthFragment = /* glsl */ `
uniform sampler2D uDay;
uniform float uReady;
uniform vec3 uSun;          // view-space light direction
varying vec3 vObj;
varying vec3 vNormalView;
varying vec3 vViewPos;
const float PI = 3.14159265;
void main() {
  vec3 n = normalize(vObj);
  float lat = asin(clamp(n.y, -1.0, 1.0));
  float lon = atan(n.x, n.z);
  vec2 uv = vec2(lon / (2.0 * PI) + 0.5, lat / PI + 0.5);
  vec3 tex = uReady > 0.5 ? texture2D(uDay, uv).rgb : vec3(0.02, 0.05, 0.08);

  // Separate ocean from land so oceans can go deep and dark.
  float ocean = smoothstep(0.02, 0.14, tex.b - max(tex.r, tex.g) * 0.9);
  float luma = dot(tex, vec3(0.299, 0.587, 0.114));
  vec3 land = mix(vec3(luma), tex, 0.45) * vec3(0.72, 0.86, 1.0);
  land = pow(land, vec3(1.18)) * 0.95;
  vec3 sea = mix(vec3(0.008, 0.03, 0.06), tex * vec3(0.25, 0.42, 0.62), 0.55);
  vec3 col = mix(land, sea, ocean);

  vec3 N = normalize(vNormalView);
  vec3 V = normalize(-vViewPos);
  float diff = dot(N, normalize(uSun));
  float day = smoothstep(-0.25, 0.55, diff);
  col *= 0.12 + 0.82 * day;

  // Soft specular glint on the water.
  vec3 H = normalize(normalize(uSun) + V);
  col += vec3(0.35, 0.55, 0.75) * pow(max(dot(N, H), 0.0), 60.0) * ocean * day * 0.6;

  // Atmospheric scattering toward the limb.
  float fres = pow(1.0 - max(dot(N, V), 0.0), 2.4);
  col = mix(col, vec3(0.45, 0.72, 1.0), fres * 0.75 * (0.35 + 0.65 * day));
  gl_FragColor = vec4(col, 1.0);
}
`;

export const cloudFragment =
  NOISE +
  /* glsl */ `
uniform float uTime;
uniform vec3 uSun;
varying vec3 vObj;
varying vec3 vNormalView;
varying vec3 vViewPos;
void main() {
  vec3 p = normalize(vObj);
  // Clouds drift slowly eastward; bands thin out near the equator.
  float c = fbm(p * 2.6 + vec3(uTime * 0.004, 0.0, -uTime * 0.003));
  c += 0.35 * fbm(p * 7.0 - vec3(uTime * 0.006));
  float cover = smoothstep(0.78, 1.08, c);
  vec3 N = normalize(vNormalView);
  float light = 0.25 + 0.85 * smoothstep(-0.2, 0.6, dot(N, normalize(uSun)));
  float fres = 1.0 - max(dot(N, normalize(-vViewPos)), 0.0);
  float a = cover * 0.62 * (1.0 - fres * 0.5);
  gl_FragColor = vec4(vec3(0.86, 0.92, 0.98) * light, a);
}
`;

export const atmosphereVertex = /* glsl */ `
varying vec3 vNormalView;
varying vec3 vViewPos;
void main() {
  vNormalView = normalize(normalMatrix * normal);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewPos = mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

// Back-faced shell: glow that falls off away from the planet's limb.
export const atmosphereFragment = /* glsl */ `
varying vec3 vNormalView;
varying vec3 vViewPos;
void main() {
  vec3 V = normalize(-vViewPos);
  float d = dot(normalize(vNormalView), V);
  float glow = pow(clamp(0.62 + d, 0.0, 1.0), 7.0);
  gl_FragColor = vec4(vec3(0.45, 0.72, 1.0) * glow * 1.6, glow);
}
`;

// Heat prisms (instanced). Top faces bright, sides darker, gentle pulse.
export const prismVertex = /* glsl */ `
uniform float uTime;
varying vec3 vColor;
varying float vTop;
void main() {
  vColor = instanceColor;
  vTop = step(0.5, normal.y);
  vec4 world = modelMatrix * instanceMatrix * vec4(position, 1.0);
  float phase = instanceMatrix[3].x * 40.0 + instanceMatrix[3].y * 30.0;
  vColor *= 0.9 + 0.12 * sin(uTime * 1.4 + phase);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

export const prismFragment = /* glsl */ `
varying vec3 vColor;
varying float vTop;
void main() {
  vec3 c = vColor * mix(0.5, 1.25, vTop);
  gl_FragColor = vec4(c, 0.92);
}
`;

// City light beams: fade toward the top.
export const beamVertex = /* glsl */ `
varying float vH;
void main() {
  vH = position.y + 0.5;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const beamFragment = /* glsl */ `
uniform vec3 uColor;
uniform float uStrength;
varying float vH;
void main() {
  float a = (1.0 - vH) * uStrength;
  gl_FragColor = vec4(uColor * (1.2 + (1.0 - vH)), a);
}
`;
