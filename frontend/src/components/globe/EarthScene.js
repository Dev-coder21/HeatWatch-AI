import * as THREE from "three";
import { geoContains } from "d3-geo";
import { mesh } from "topojson-client";
import world from "world-atlas/countries-50m.json";

import india from "../../assets/india_boundary.json";
import {
  atmosphereFragment,
  atmosphereVertex,
  beamFragment,
  beamVertex,
  cloudFragment,
  earthFragment,
  prismFragment,
  prismVertex,
  surfaceVertex,
} from "./shaders";

const DEG = Math.PI / 180;
const INDIA = india.features[0];
const TEXTURE_URL = "/textures/earth-blue-marble.jpg";

// Where the camera parks: India's centre, viewed from slightly south so the
// planet's upper limb and the horizon stay in frame.
export const HOME = { lat: 22, lon: 80.5 };
const CAM = { pos: new THREE.Vector3(0, -0.9, 3.5), look: new THREE.Vector3(0, 0.42, 0) };
// Degrees by which the parked target sits "up the planet" from the point nearest the camera,
// so India appears mid-frame with the horizon above it.
const TILT = 16;
const CAM_LAT = Math.asin(CAM.pos.clone().normalize().y) / DEG;

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;

export function toVec(lat, lon, r = 1) {
  const la = lat * DEG;
  const lo = lon * DEG;
  return new THREE.Vector3(r * Math.cos(la) * Math.sin(lo), r * Math.sin(la), r * Math.cos(la) * Math.cos(lo));
}

function toLatLon(v) {
  const n = v.clone().normalize();
  return { lat: Math.asin(n.y) / DEG, lon: Math.atan2(n.x, n.z) / DEG };
}

/* Hexagonal lattice of points inside India (rows offset by half a step). */
function hexLattice(step) {
  const pts = [];
  const rowStep = step * 0.866;
  let row = 0;
  for (let lat = 6.4; lat <= 37.4; lat += rowStep, row += 1) {
    const offset = row % 2 ? step / 2 : 0;
    // Keep the lattice roughly equal-area at higher latitudes.
    const lonStep = step / Math.cos(lat * DEG);
    for (let lon = 67.8 + offset; lon <= 97.6; lon += lonStep) {
      if (geoContains(INDIA, [lon, lat])) pts.push([lon, lat]);
    }
  }
  return pts;
}

function linesFromGeometry(geometry, radius) {
  const positions = [];
  const addRing = (ring) => {
    for (let i = 0; i < ring.length - 1; i += 1) {
      const a = toVec(ring[i][1], ring[i][0], radius);
      const b = toVec(ring[i + 1][1], ring[i + 1][0], radius);
      positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  };
  if (geometry.type === "MultiLineString") geometry.coordinates.forEach(addRing);
  if (geometry.type === "LineString") addRing(geometry.coordinates);
  if (geometry.type === "Polygon") geometry.coordinates.forEach(addRing);
  if (geometry.type === "MultiPolygon") geometry.coordinates.forEach((poly) => poly.forEach(addRing));
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return g;
}

export default class EarthScene {
  constructor(container, labelLayer, { onPickCity, onPickPoint, onHover }) {
    this.container = container;
    this.labelLayer = labelLayer;
    this.callbacks = { onPickCity, onPickPoint, onHover };
    this.reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    this.view = { lat: HOME.lat, lon: HOME.lon, dist: 1, lookLift: 1, tilt: TILT };
    this.anim = null;
    this.drag = null;
    this.cities = [];
    this.markers = new Map();
    this.hoverSlug = null;
    this.selectedSlug = null;
    this.clock = new THREE.Clock();
    this.frame = 0;
    this.visible = true;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x020407, 1);
    this.renderer.domElement.className = "earth-canvas";
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.01, 200);
    this.globe = new THREE.Group();
    this.scene.add(this.globe);
    this.sun = new THREE.Vector3(-0.55, 0.55, 0.62).normalize();

    this.buildStars();
    this.buildEarth();
    this.buildBorders();
    this.buildHeatField();
    this.raycaster = new THREE.Raycaster();

    // Entrance: the planet approaches from deep space.
    if (!this.reduceMotion) {
      this.view.dist = 6.5;
      this.view.lon = HOME.lon - 70;
      this.view.lat = HOME.lat - 10;
      this.animateTo({ lat: HOME.lat, lon: HOME.lon, dist: 1, lookLift: 1, tilt: TILT }, 3600, easeInOut);
    }

    this.bindEvents();
    this.resize();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.io = new IntersectionObserver(([e]) => {
      this.visible = e?.isIntersecting ?? true;
      this.visible ? this.start() : this.stop();
    });
    this.io.observe(container);
    this.onVis = () => (document.hidden ? this.stop() : this.start());
    document.addEventListener("visibilitychange", this.onVis);
    this.start();
  }

  /* ----------------------------- build ----------------------------- */

  buildStars() {
    const n = 1800;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i += 1) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(60 + Math.random() * 20);
      pos.set([v.x, v.y, v.z], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.stars = new THREE.Points(
      g,
      new THREE.PointsMaterial({ color: 0x9fb8d6, size: 0.12, sizeAttenuation: true, transparent: true, opacity: 0.55, depthWrite: false }),
    );
    this.scene.add(this.stars);
  }

  buildEarth() {
    const uniforms = { uDay: { value: null }, uReady: { value: 0 }, uSun: { value: this.sun.clone() } };
    const loader = new THREE.TextureLoader();
    this.texture = loader.load(TEXTURE_URL, (tex) => {
      tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
      tex.wrapS = THREE.RepeatWrapping;
      tex.needsUpdate = true;
      uniforms.uDay.value = tex;
      uniforms.uReady.value = 1;
    });
    const sphere = new THREE.SphereGeometry(1, 160, 120);
    this.earth = new THREE.Mesh(sphere, new THREE.ShaderMaterial({ uniforms, vertexShader: surfaceVertex, fragmentShader: earthFragment }));
    this.globe.add(this.earth);

    this.cloudUniforms = { uTime: { value: 0 }, uSun: uniforms.uSun };
    this.clouds = new THREE.Mesh(
      new THREE.SphereGeometry(1.012, 128, 96),
      new THREE.ShaderMaterial({
        uniforms: this.cloudUniforms,
        vertexShader: surfaceVertex,
        fragmentShader: cloudFragment,
        transparent: true,
        depthWrite: false,
      }),
    );
    this.globe.add(this.clouds);

    this.atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.045, 128, 96),
      new THREE.ShaderMaterial({
        vertexShader: atmosphereVertex,
        fragmentShader: atmosphereFragment,
        side: THREE.BackSide,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    this.scene.add(this.atmosphere);
  }

  buildBorders() {
    const borders = linesFromGeometry(mesh(world, world.objects.countries, (a, b) => a !== b), 1.0016);
    this.borderLines = new THREE.LineSegments(borders, new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.22, depthWrite: false }));
    this.globe.add(this.borderLines);

    // India: a bright core line with soft additive halos.
    const indiaGroup = new THREE.Group();
    [
      [1.0022, 0.95, 0xe0f6ff],
      [1.0032, 0.35, 0x38bdf8],
      [1.0045, 0.18, 0x38bdf8],
    ].forEach(([r, opacity, color]) => {
      indiaGroup.add(
        new THREE.LineSegments(
          linesFromGeometry(INDIA.geometry, r),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }),
        ),
      );
    });
    this.globe.add(indiaGroup);
  }

  buildHeatField() {
    const step = 0.36;
    this.cells = hexLattice(step);
    const radius = step * DEG * 0.43;
    const geo = new THREE.CylinderGeometry(radius, radius, 1, 6, 1, false);
    geo.translate(0, 0.5, 0);
    this.prismUniforms = { uTime: { value: 0 } };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.prismUniforms,
      vertexShader: prismVertex,
      fragmentShader: prismFragment,
      transparent: true,
    });
    this.prisms = new THREE.InstancedMesh(geo, mat, this.cells.length);
    this.prisms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cellBase = this.cells.map(([lon, lat]) => {
      const n = toVec(lat, lon).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);
      return { pos: n.clone().multiplyScalar(1.0008), q };
    });
    this.heights = new Float32Array(this.cells.length);
    this.targetHeights = new Float32Array(this.cells.length);
    const color = new THREE.Color(0x0a1520);
    for (let i = 0; i < this.cells.length; i += 1) this.prisms.setColorAt(i, color);
    this.writePrisms();
    this.globe.add(this.prisms);
  }

  writePrisms() {
    const m = new THREE.Matrix4();
    const s = new THREE.Vector3();
    for (let i = 0; i < this.cells.length; i += 1) {
      const b = this.cellBase[i];
      s.set(1, Math.max(0.0004, this.heights[i]), 1);
      m.compose(b.pos, b.q, s);
      this.prisms.setMatrixAt(i, m);
    }
    this.prisms.instanceMatrix.needsUpdate = true;
  }

  /* ----------------------------- data ------------------------------ */

  /** values: per-cell numbers; colorOf(v) -> css colour; heightOf(v) -> 0..1 */
  setField(values, colorOf, heightOf) {
    const c = new THREE.Color();
    for (let i = 0; i < this.cells.length; i += 1) {
      const v = values?.[i];
      if (!Number.isFinite(v)) {
        this.targetHeights[i] = 0.0005;
        c.set(0x0a1520);
      } else {
        this.targetHeights[i] = 0.002 + Math.max(0, Math.min(1.2, heightOf(v))) * 0.03;
        c.set(colorOf(v));
      }
      this.prisms.setColorAt(i, c);
    }
    this.prisms.instanceColor.needsUpdate = true;
    if (this.reduceMotion) {
      this.heights.set(this.targetHeights);
      this.writePrisms();
    }
  }

  cellCoordinates() {
    return this.cells;
  }

  /** cities: [{slug, name, lat, lon, risk, color, value, label}] */
  setCities(cities) {
    this.cities = cities;
    const keep = new Set(cities.map((c) => c.slug));
    for (const [slug, mk] of this.markers) {
      if (!keep.has(slug)) {
        this.globe.remove(mk.group);
        mk.label.remove();
        this.markers.delete(slug);
      }
    }
    for (const city of cities) {
      let mk = this.markers.get(city.slug);
      if (!mk) mk = this.createMarker(city);
      mk.city = city;
      const height = 0.04 + (city.risk ?? 0) * 0.0016;
      mk.height = height;
      mk.beam.scale.set(1, height, 1);
      mk.beam.position.set(0, height / 2, 0);
      mk.beamMat.uniforms.uColor.value.set(city.color);
      mk.ring.material.color.set(city.color);
      mk.dot.material.color.set(city.color);
      mk.label.querySelector(".gl-name").textContent = city.name;
      mk.label.querySelector(".gl-val").textContent = city.label;
      mk.label.style.setProperty("--c", city.color);
    }
  }

  createMarker(city) {
    const group = new THREE.Group();
    const n = toVec(city.lat, city.lon).normalize();
    group.position.copy(n.clone().multiplyScalar(1.001));
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);

    const beamMat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color(city.color) }, uStrength: { value: 0.85 } },
      vertexShader: beamVertex,
      fragmentShader: beamFragment,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.0018, 0.0026, 1, 10, 1, true), beamMat);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.008, 0.0098, 48),
      new THREE.MeshBasicMaterial({ color: city.color, transparent: true, opacity: 0.8, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.0015;
    const dot = new THREE.Mesh(
      new THREE.CircleGeometry(0.0042, 24),
      new THREE.MeshBasicMaterial({ color: city.color, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    dot.rotation.x = -Math.PI / 2;
    dot.position.y = 0.002;
    group.add(beam, ring, dot);
    this.globe.add(group);

    const label = document.createElement("button");
    label.type = "button";
    label.className = "gl-label";
    label.innerHTML = '<span class="gl-name"></span><span class="gl-val"></span>';
    label.addEventListener("click", (e) => {
      e.stopPropagation();
      this.callbacks.onPickCity?.(this.markers.get(city.slug).city);
    });
    label.addEventListener("pointerenter", () => this.setHover(city.slug));
    label.addEventListener("pointerleave", () => this.setHover(null));
    this.labelLayer.appendChild(label);

    const mk = { group, beam, beamMat, ring, dot, label, city, height: 0.05, phase: Math.random() * 2 };
    this.markers.set(city.slug, mk);
    return mk;
  }

  setHover(slug) {
    this.hoverSlug = slug;
    const mk = slug ? this.markers.get(slug) : null;
    this.callbacks.onHover?.(mk ? mk.city : null);
  }

  /* ----------------------------- camera ---------------------------- */

  animateTo(to, duration, ease = easeInOut) {
    return new Promise((resolve) => {
      if (this.reduceMotion) {
        Object.assign(this.view, to);
        resolve();
        return;
      }
      this.anim = { from: { ...this.view }, to: { ...this.view, ...to }, start: performance.now(), duration, ease, resolve };
    });
  }

  /** Planet -> region -> city: centre the city and descend toward it. */
  flyTo(lat, lon, slug) {
    this.selectedSlug = slug || null;
    return this.animateTo({ lat, lon, dist: 0.05, lookLift: 0, tilt: 0 }, 1800, (t) => t * t * (3 - 2 * t));
  }

  home() {
    return this.animateTo({ lat: HOME.lat, lon: HOME.lon, dist: 1, lookLift: 1, tilt: TILT }, 1200);
  }

  applyView(time) {
    const v = this.view;
    // Slow cinematic drift around the parked view.
    const drift = this.reduceMotion || this.drag || this.anim ? 0 : 1;
    const lon = v.lon + drift * Math.sin(time * 0.045) * 3;
    const lat = v.lat + drift * Math.sin(time * 0.031) * 1.2;
    this.globe.rotation.set((lat - CAM_LAT - v.tilt) * DEG, -lon * DEG, 0, "XYZ");

    // Altitude scales along the camera's line to the planet; the look-at point
    // blends from above the horizon (home view) down to the surface (city dive).
    const dir = CAM.pos.clone().normalize();
    const altitude = 1 + (CAM.pos.length() - 1) * v.dist;
    this.camera.position.copy(dir.clone().multiplyScalar(altitude));
    this.camera.lookAt(dir.clone().lerp(CAM.look, v.lookLift));
  }

  /* ----------------------------- loop ------------------------------ */

  start() {
    if (!this.frame && this.visible && !document.hidden) this.frame = requestAnimationFrame((t) => this.render(t));
  }

  stop() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
  }

  render(now) {
    this.frame = 0;
    const time = this.clock.getElapsedTime();

    if (this.anim) {
      const k = Math.min(1, (now - this.anim.start) / this.anim.duration);
      const e = this.anim.ease(Math.max(0, k));
      for (const key of Object.keys(this.anim.to)) this.view[key] = lerp(this.anim.from[key], this.anim.to[key], e);
      if (k >= 1) {
        const done = this.anim.resolve;
        this.anim = null;
        done?.();
      }
    }
    this.applyView(time);

    this.cloudUniforms.uTime.value = this.reduceMotion ? 0 : time;
    this.prismUniforms.uTime.value = this.reduceMotion ? 0 : time;
    this.clouds.rotation.y = this.reduceMotion ? 0 : time * 0.002;

    // Ease prism heights toward their targets (data changes animate).
    let moving = false;
    for (let i = 0; i < this.heights.length; i += 1) {
      const d = this.targetHeights[i] - this.heights[i];
      if (Math.abs(d) > 1e-5) {
        this.heights[i] += d * 0.08;
        moving = true;
      }
    }
    if (moving) this.writePrisms();

    this.updateMarkers(time);
    this.renderer.render(this.scene, this.camera);
    if (this.visible && !document.hidden) this.frame = requestAnimationFrame((t) => this.render(t));
  }

  updateMarkers(time) {
    const w = this.width;
    const h = this.height;
    const camDir = new THREE.Vector3();
    const top = new THREE.Vector3();
    const normal = new THREE.Vector3();
    this.globe.updateMatrixWorld();
    for (const mk of this.markers.values()) {
      const active = mk.city.slug === this.hoverSlug || mk.city.slug === this.selectedSlug;
      const pulse = this.reduceMotion ? 0.5 : (time * 0.6 + mk.phase) % 1;
      mk.ring.scale.setScalar(1 + pulse * 2.6 + (active ? 0.6 : 0));
      mk.ring.material.opacity = (1 - pulse) * 0.8;
      mk.beamMat.uniforms.uStrength.value = active ? 1.4 : 0.85;
      mk.beam.scale.x = mk.beam.scale.z = active ? 1.8 : 1;

      // Screen position of the beam top for the HTML label.
      top.set(0, mk.height + 0.006, 0).applyMatrix4(mk.group.matrixWorld);
      normal.copy(mk.group.position).normalize().transformDirection(this.globe.matrixWorld);
      camDir.copy(this.camera.position).sub(top).normalize();
      const facing = normal.dot(camDir);
      const p = top.clone().project(this.camera);
      const x = (p.x * 0.5 + 0.5) * w;
      const y = (-p.y * 0.5 + 0.5) * h;
      const shown = facing > 0.12 && p.z < 1;
      mk.label.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
      mk.label.style.opacity = shown ? String(Math.min(1, (facing - 0.12) * 6)) : "0";
      mk.label.style.pointerEvents = shown ? "auto" : "none";
      mk.label.classList.toggle("active", active);
    }
  }

  /* ----------------------------- input ----------------------------- */

  bindEvents() {
    const el = this.renderer.domElement;
    this.handlers = {
      down: (e) => {
        if (this.anim?.resolve && this.view.dist < 1) return; // mid fly-to
        this.anim = null;
        this.drag = { x: e.clientX, y: e.clientY, lat: this.view.lat, lon: this.view.lon, moved: false };
        el.setPointerCapture(e.pointerId);
      },
      move: (e) => {
        if (!this.drag) return;
        const dx = e.clientX - this.drag.x;
        const dy = e.clientY - this.drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) this.drag.moved = true;
        const k = 0.09 * this.view.dist;
        this.view.lon = this.drag.lon - dx * k;
        this.view.lat = Math.max(-40, Math.min(60, this.drag.lat + dy * k));
      },
      up: (e) => {
        const drag = this.drag;
        this.drag = null;
        if (!drag || drag.moved) return;
        const hit = this.pick(e.clientX, e.clientY);
        if (hit) this.callbacks.onPickPoint?.(hit.lat, hit.lon);
      },
      wheel: (e) => {
        e.preventDefault();
        this.anim = null;
        this.view.dist = Math.max(0.55, Math.min(2.4, this.view.dist * Math.exp(e.deltaY * 0.0011)));
      },
      dbl: () => this.home(),
    };
    el.addEventListener("pointerdown", this.handlers.down);
    el.addEventListener("pointermove", this.handlers.move);
    el.addEventListener("pointerup", this.handlers.up);
    el.addEventListener("wheel", this.handlers.wheel, { passive: false });
    el.addEventListener("dblclick", this.handlers.dbl);
  }

  pick(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = this.raycaster.intersectObject(this.earth, false)[0];
    if (!hit) return null;
    const local = this.earth.worldToLocal(hit.point.clone());
    return toLatLon(local);
  }

  resize() {
    const { width, height } = this.container.getBoundingClientRect();
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.domElement.style.width = `${this.width}px`;
    this.renderer.domElement.style.height = `${this.height}px`;
    this.camera.aspect = this.width / this.height;
    // Portrait screens: narrow the vertical field of view so India spans the width.
    this.camera.fov = this.camera.aspect < 0.9 ? 29 : 34;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    this.stop();
    this.ro.disconnect();
    this.io.disconnect();
    document.removeEventListener("visibilitychange", this.onVis);
    const el = this.renderer.domElement;
    el.removeEventListener("pointerdown", this.handlers.down);
    el.removeEventListener("pointermove", this.handlers.move);
    el.removeEventListener("pointerup", this.handlers.up);
    el.removeEventListener("wheel", this.handlers.wheel);
    el.removeEventListener("dblclick", this.handlers.dbl);
    for (const mk of this.markers.values()) mk.label.remove();
    this.scene.traverse((o) => {
      o.geometry?.dispose();
      if (o.material) [].concat(o.material).forEach((m) => m.dispose());
    });
    this.texture?.dispose();
    this.renderer.dispose();
    el.remove();
  }
}
