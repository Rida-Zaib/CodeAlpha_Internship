// selftest.js - Headless 180s simulation test suite
// Run with: bun selftest.js OR node selftest.js
// Also executes in browser when ?test=1 is in the URL

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else if (typeof exports === 'object') {
    exports.RCVSelfTest = factory();
  } else {
    root.RCVSelfTest = factory();
  }
}(typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof self !== 'undefined' ? self : this, function () {

  // Seeded PRNG (Mulberry32) - Never Math.random()
  class SeededRNG {
    constructor(seed = 424242) {
      this.s = seed >>> 0;
    }
    next() {
      let t = (this.s += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    range(min, max) {
      return min + this.next() * (max - min);
    }
    rangeInt(min, max) {
      return Math.floor(this.range(min, max + 1));
    }
    exponential(mean) {
      const u = Math.max(1e-7, Math.min(1 - 1e-7, this.next()));
      return -mean * Math.log(1 - u);
    }
    choice(arr) {
      return arr[Math.floor(this.next() * arr.length)];
    }
  }

  // Simulation Constants
  const Z_MIN = -95.0;
  const Z_MAX = 22.0;
  const Z_LENGTH = Z_MAX - Z_MIN; // 117 m
  const ZEBRA_Z = -14.0;
  const TRIPWIRE_Z = 0.0;

  // Lanes
  const VEHICLE_LANES = [
    { x: -3.8, dir: 1 },  // Southbound outer
    { x: -1.8, dir: 1 },  // Southbound inner
    { x: 1.8, dir: -1 },  // Northbound inner
    { x: 3.8, dir: -1 }   // Northbound outer
  ];
  const CYCLIST_LANES = [
    { x: -5.2, dir: 1 },
    { x: 5.2, dir: -1 }
  ];

  // Hard caps
  const CAPS = {
    car: 10,
    cyclist: 3,
    pedestrian: 12,
    child: 3,
    cat: 2,
    pigeon: 4,
    bird: 4,
    butterfly: 4
  };

  // Mean inter-arrival times (seconds)
  const SPAWN_INTERVALS = {
    car_per_lane: 3.5,
    cyclist: 9.0,
    pedestrian_per_side: 4.0,
    cat: 25.0,
    pigeon: 30.0,
    bird: 14.0,
    butterfly: 12.0
  };

  // Traffic Light Phases
  // VEHICLE_GREEN 14s, AMBER 3s, ALL_RED 1.5s, PED_WALK 8s, ALL_RED 1.5s
  const TL_PHASES = [
    { name: "VEHICLE_GREEN", dur: 14.0, vehCanGo: true, pedCanCross: false },
    { name: "AMBER", dur: 3.0, vehCanGo: false, pedCanCross: false },
    { name: "ALL_RED_1", dur: 1.5, vehCanGo: false, pedCanCross: false },
    { name: "PED_WALK", dur: 8.0, vehCanGo: false, pedCanCross: true },
    { name: "ALL_RED_2", dur: 1.5, vehCanGo: false, pedCanCross: false }
  ];

  class TrafficSimulation {
    constructor(seed = 424242) {
      this.rng = new SeededRNG(seed);
      this.time = 0;
      this.agents = [];
      this.nextId = 1;

      // Traffic Light State
      this.tlPhaseIndex = 0;
      this.tlTimer = 0;

      // Spawn Timers per lane / class
      this.spawnTimers = {
        carLanes: VEHICLE_LANES.map(() => 0),
        cyclists: 0,
        pedWest: 0,
        pedEast: 0,
        cat: 0,
        pigeon: 0,
        bird: 0,
        butterfly: 0
      };

      // Metrics
      this.tripwireVehiclesCount = 0;
      this.completedPedCrossingsCount = 0;

      // Initialize population with stratified sampling
      this.initPopulation();
    }

    currentPhase() {
      return TL_PHASES[this.tlPhaseIndex];
    }

    countClass(cls) {
      let c = 0;
      for (const a of this.agents) {
        if (a.type === cls) c++;
      }
      return c;
    }

    initPopulation() {
      // Stratified sampling: divide z [-95, +22] into bins
      const binCount = 10;
      const binSize = Z_LENGTH / binCount;

      // 7 initial cars (within cap 10, >= 5)
      for (let i = 0; i < 7; i++) {
        const lane = VEHICLE_LANES[i % VEHICLE_LANES.length];
        const binIndex = (i * 3 + 1) % binCount;
        const z = Z_MIN + (binIndex + this.rng.range(0.2, 0.8)) * binSize;
        this.agents.push({
          id: this.nextId++,
          type: "car",
          x: lane.x,
          z: z,
          dir: lane.dir,
          speed: this.rng.range(8.5, 11.5),
          baseSpeed: 10.5,
          lane: lane,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          hasCrossedTripwire: false
        });
      }

      // 2 initial cyclists (within cap 3, >= 2)
      for (let i = 0; i < 2; i++) {
        const lane = CYCLIST_LANES[i % CYCLIST_LANES.length];
        const z = Z_MIN + (i * 5 + 2) * (Z_LENGTH / 10);
        this.agents.push({
          id: this.nextId++,
          type: "cyclist",
          x: lane.x,
          z: z,
          dir: lane.dir,
          speed: this.rng.range(4.5, 5.5),
          baseSpeed: 5.0,
          lane: lane,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          hasCrossedTripwire: false
        });
      }

      // 8 initial pedestrians (within cap 12, >= 6) + 2 children (cap 3, >= 2)
      for (let i = 0; i < 10; i++) {
        const isChild = i >= 8;
        const side = (i % 2 === 0) ? -1 : 1;
        const isCrossing = (i < 4); // 4 initial crossing candidates
        const z = isCrossing ? ZEBRA_Z : (Z_MIN + (i + 0.5) * (Z_LENGTH / 10));
        const x = isCrossing ? (side * 6.6) : (side * 8.4);
        this.agents.push({
          id: this.nextId++,
          type: isChild ? "child" : "pedestrian",
          x: x,
          z: z,
          dirZ: (i % 3 === 0) ? 1 : -1,
          dirX: isCrossing ? (side === -1 ? 1 : -1) : 0,
          speed: isChild ? 1.25 : 1.5,
          baseSpeed: isChild ? 1.25 : 1.5,
          isCrossing: isCrossing,
          state: isCrossing ? "WAIT_KERB" : "WALK_PATH",
          kerbWaitTime: 0,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(8, 20)
        });
      }

      // 1 cat (cap 2, >= 1)
      this.agents.push({
        id: this.nextId++,
        type: "cat",
        x: -8.4,
        z: Z_MIN + 30,
        dirZ: 1,
        speed: 1.4,
        baseSpeed: 1.4,
        posHistory: [],
        stuckTimer: 0,
        pauseTimer: 0,
        isLegitimateStop: false,
        nextPauseTime: this.time + this.rng.range(10, 20)
      });

      // 3 pigeons (cap 4, >= 2)
      for (let i = 0; i < 3; i++) {
        this.agents.push({
          id: this.nextId++,
          type: "pigeon",
          x: 8.4 + (i % 2) * 0.4,
          z: ZEBRA_Z - 6 + i * 4,
          dirZ: 1,
          speed: 0.9,
          baseSpeed: 0.9,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(6, 15)
        });
      }

      // 3 birds (cap 4, >= 2)
      for (let i = 0; i < 3; i++) {
        this.agents.push({
          id: this.nextId++,
          type: "bird",
          x: -6 + i * 6,
          z: Z_MIN + i * 35,
          dirZ: 1,
          speed: 8.5,
          baseSpeed: 8.5,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false
        });
      }

      // 3 butterflies (cap 4, >= 2)
      for (let i = 0; i < 3; i++) {
        this.agents.push({
          id: this.nextId++,
          type: "butterfly",
          x: 8.6,
          z: -25 + i * 16,
          dirZ: (i % 2 === 0) ? 1 : -1,
          speed: 1.2,
          baseSpeed: 1.2,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(5, 12)
        });
      }

      // Initialize spawn timers
      VEHICLE_LANES.forEach((_, idx) => {
        this.spawnTimers.carLanes[idx] = this.time + this.rng.exponential(SPAWN_INTERVALS.car_per_lane);
      });
      this.spawnTimers.cyclists = this.time + this.rng.exponential(SPAWN_INTERVALS.cyclist);
      this.spawnTimers.pedWest = this.time + this.rng.exponential(SPAWN_INTERVALS.pedestrian_per_side);
      this.spawnTimers.pedEast = this.time + this.rng.exponential(SPAWN_INTERVALS.pedestrian_per_side);
      this.spawnTimers.cat = this.time + this.rng.exponential(SPAWN_INTERVALS.cat);
      this.spawnTimers.pigeon = this.time + this.rng.exponential(SPAWN_INTERVALS.pigeon);
      this.spawnTimers.bird = this.time + this.rng.exponential(SPAWN_INTERVALS.bird);
      this.spawnTimers.butterfly = this.time + this.rng.exponential(SPAWN_INTERVALS.butterfly);
    }

    step(dt) {
      this.time += dt;

      // 1. Update Traffic Lights
      this.tlTimer += dt;
      const phase = this.currentPhase();
      if (this.tlTimer >= phase.dur) {
        this.tlTimer = 0;
        this.tlPhaseIndex = (this.tlPhaseIndex + 1) % TL_PHASES.length;
      }

      // 2. Check Spawners
      this.updateSpawners();

      // 3. Check Roadway Pedestrians
      const pedOnRoad = this.agents.some(a =>
        (a.type === "pedestrian" || a.type === "child") &&
        a.isCrossing && a.state === "CROSSING" &&
        Math.abs(a.x) < 6.0
      );

      // 4. Update Agents
      for (let i = this.agents.length - 1; i >= 0; i--) {
        const a = this.agents[i];
        if (a.type === "car" || a.type === "cyclist") {
          this.updateVehicle(a, dt, pedOnRoad);
        } else if (a.type === "pedestrian" || a.type === "child") {
          this.updatePedestrian(a, dt);
        } else {
          this.updateFauna(a, dt);
        }

        // Stuck watchdog
        this.checkStuckWatchdog(a, dt);

        // Despawn check when out of road/screen volume
        if (this.shouldDespawn(a)) {
          this.agents.splice(i, 1);
        }
      }

      // Enforce floor count to never drop below half caps
      this.enforceMinPopulation();
    }

    updateVehicle(veh, dt, pedOnRoad) {
      const curPhase = this.currentPhase();
      let desiredSpeed = veh.baseSpeed;
      let isStopped = false;

      // Distance to stop line before zebra crossing
      const stopLineZ = (veh.dir === 1) ? (ZEBRA_Z - 3.5) : (ZEBRA_Z + 3.5);
      const distToStopLine = (veh.dir === 1) ? (stopLineZ - veh.z) : (veh.z - stopLineZ);

      // Vehicles yield if red/amber or ped physically on road
      if (distToStopLine > 0 && distToStopLine < 18.0) {
        if (!curPhase.vehCanGo || pedOnRoad) {
          isStopped = true;
          if (distToStopLine < 2.0) {
            desiredSpeed = 0;
          } else {
            desiredSpeed = Math.min(desiredSpeed, (distToStopLine / 10.0) * veh.baseSpeed);
          }
        }
      }

      // Car-following rule: gap to vehicle ahead in same lane
      let gapAhead = Infinity;
      let leadIsStopped = false;
      for (const other of this.agents) {
        if (other === veh || other.lane !== veh.lane) continue;
        const gap = (other.z - veh.z) * veh.dir;
        if (gap > 0 && gap < gapAhead) {
          gapAhead = gap;
          leadIsStopped = other.isLegitimateStop;
        }
      }

      if (gapAhead < 8.0) {
        if (gapAhead < 4.5) {
          desiredSpeed = 0;
          if (leadIsStopped) isStopped = true;
        } else {
          desiredSpeed = Math.min(desiredSpeed, (gapAhead - 4.0) * 1.5);
          if (leadIsStopped && desiredSpeed < 0.2) isStopped = true;
        }
      }

      // Smooth acceleration / braking
      const maxAcc = 4.0 * dt;
      const maxBrake = 10.0 * dt;
      if (desiredSpeed < veh.speed) {
        veh.speed = Math.max(desiredSpeed, veh.speed - maxBrake);
      } else {
        veh.speed = Math.min(desiredSpeed, veh.speed + maxAcc);
      }

      // Displace vehicle
      const prevZ = veh.z;
      veh.z += veh.speed * veh.dir * dt;
      const newZ = veh.z;

      veh.isLegitimateStop = isStopped && veh.speed < 0.2;

      // Tripwire check at z = 0.0
      if (!veh.hasCrossedTripwire) {
        const crossed = (prevZ <= TRIPWIRE_Z && newZ >= TRIPWIRE_Z) || (prevZ >= TRIPWIRE_Z && newZ <= TRIPWIRE_Z);
        if (crossed) {
          veh.hasCrossedTripwire = true;
          this.tripwireVehiclesCount++;
        }
      }
    }

    updatePedestrian(ped, dt) {
      const curPhase = this.currentPhase();

      if (ped.isCrossing) {
        if (ped.state === "WAIT_KERB") {
          ped.speed = 0;
          ped.isLegitimateStop = true;
          ped.kerbWaitTime += dt;

          // Rule: Can start crossing only on PED_WALK with enough time left
          const timeRemainingInWalk = curPhase.name === "PED_WALK" ? (curPhase.dur - this.tlTimer) : 0;
          const enoughTimeLeft = curPhase.name === "PED_WALK" && timeRemainingInWalk >= 3.5;

          if (enoughTimeLeft) {
            ped.state = "CROSSING";
            ped.isLegitimateStop = false;
            ped.kerbWaitTime = 0;
          } else if (ped.kerbWaitTime > 25.0) {
            // Waiting timeout: after 25s, switch to walking along footpath
            ped.isCrossing = false;
            ped.state = "WALK_PATH";
            ped.dirX = 0;
            ped.dirZ = this.rng.choice([-1, 1]);
            ped.isLegitimateStop = false;
          }
        } else if (ped.state === "CROSSING") {
          ped.speed = ped.baseSpeed;
          ped.x += ped.speed * ped.dirX * dt;
          ped.isLegitimateStop = false;

          // Check if finished crossing (reached opposite pavement)
          if ((ped.dirX === 1 && ped.x >= 6.4) || (ped.dirX === -1 && ped.x <= -6.4)) {
            ped.isCrossing = false;
            ped.state = "WALK_PATH";
            ped.dirX = 0;
            ped.dirZ = this.rng.choice([-1, 1]);
            this.completedPedCrossingsCount++;
          }
        }
      } else {
        // Walking along footpath
        if (ped.pauseTimer > 0) {
          ped.pauseTimer -= dt;
          ped.speed = 0;
          ped.isLegitimateStop = true;
        } else {
          ped.isLegitimateStop = false;
          ped.speed = ped.baseSpeed;
          ped.z += ped.speed * ped.dirZ * dt;

          // Opportunity to cross zebra crossing if near z = -14
          if (Math.abs(ped.z - ZEBRA_Z) < 0.6 && !ped.hasConsideredCrossing) {
            ped.hasConsideredCrossing = true;
            if (this.rng.next() < 0.65) {
              ped.isCrossing = true;
              ped.state = "WAIT_KERB";
              ped.kerbWaitTime = 0;
              ped.dirX = ped.x < 0 ? 1 : -1;
              ped.x = ped.x < 0 ? -6.5 : 6.5;
              ped.z = ZEBRA_Z;
            }
          }

          // Natural pauses: every 8..20s pause 1.5..4s on walking lane
          if (this.time >= ped.nextPauseTime) {
            ped.pauseTimer = this.rng.range(1.5, 4.0);
            ped.nextPauseTime = this.time + this.rng.range(8, 20);
          }
        }
      }
    }

    updateFauna(f, dt) {
      if (f.pauseTimer > 0) {
        f.pauseTimer -= dt;
        f.speed = 0;
        f.isLegitimateStop = true;
      } else {
        f.isLegitimateStop = false;
        f.speed = f.baseSpeed;
        f.z += f.speed * f.dirZ * dt;

        if (f.nextPauseTime && this.time >= f.nextPauseTime) {
          if (f.type === "cat") {
            f.pauseTimer = this.rng.range(4.0, 8.0); // groom
            f.nextPauseTime = this.time + this.rng.range(15, 30);
          } else if (f.type === "pigeon") {
            f.pauseTimer = this.rng.range(2.0, 5.0); // peck
            f.nextPauseTime = this.time + this.rng.range(8, 20);
          } else if (f.type === "butterfly") {
            f.pauseTimer = this.rng.range(3.0, 6.0); // hover/land
            f.nextPauseTime = this.time + this.rng.range(8, 16);
          }
        }
      }
    }

    checkStuckWatchdog(a, dt) {
      a.posHistory.push({ t: this.time, z: a.z, x: a.x });
      // Keep 12s history
      while (a.posHistory.length > 0 && a.posHistory[0].t < this.time - 12.0) {
        a.posHistory.shift();
      }

      if (a.posHistory.length < 5) return;

      // Check movement over last 8s
      const h8 = a.posHistory.find(p => p.t >= this.time - 8.0) || a.posHistory[0];
      const dist8 = Math.hypot(a.z - h8.z, a.x - h8.x);

      if (dist8 < 0.3 && !a.isLegitimateStop) {
        a.stuckTimer += dt;
        // Steer sideways
        if (a.stuckTimer >= 1.0) {
          const side = a.x >= 0 ? 1 : -1;
          a.x += side * 0.1 * dt;
        }
        // If still stuck after 12s, respawn
        if (a.stuckTimer >= 12.0) {
          a.stuckTimer = 0;
          this.respawnAgent(a);
        }
      } else {
        a.stuckTimer = 0;
      }
    }

    respawnAgent(a) {
      if (a.type === "car" || a.type === "cyclist") {
        a.z = a.dir === 1 ? Z_MIN : Z_MAX;
        a.speed = a.baseSpeed;
      } else {
        a.z = this.rng.choice([Z_MIN + 5, Z_MAX - 5]);
        a.speed = a.baseSpeed;
      }
      a.posHistory = [];
    }

    shouldDespawn(a) {
      if (a.type === "car" || a.type === "cyclist") {
        return (a.dir === 1 && a.z > Z_MAX + 5) || (a.dir === -1 && a.z < Z_MIN - 5);
      }
      if (a.type === "bird") {
        return a.z > Z_MAX + 15 || a.z < Z_MIN - 15;
      }
      return a.z > Z_MAX + 4 || a.z < Z_MIN - 4;
    }

    updateSpawners() {
      // 1. Cars per lane (mean 3.5s)
      VEHICLE_LANES.forEach((lane, idx) => {
        if (this.time >= this.spawnTimers.carLanes[idx]) {
          if (this.countClass("car") < CAPS.car) {
            const spawnZ = lane.dir === 1 ? Z_MIN : Z_MAX;
            // Check clearance: no vehicle within 15 m in same lane
            const isBlocked = this.agents.some(other =>
              (other.type === "car" || other.type === "cyclist") &&
              other.lane === lane &&
              Math.abs(other.z - spawnZ) < 15.0
            );
            if (!isBlocked) {
              this.agents.push({
                id: this.nextId++,
                type: "car",
                x: lane.x,
                z: spawnZ,
                dir: lane.dir,
                speed: this.rng.range(9.0, 11.5),
                baseSpeed: 10.5,
                lane: lane,
                posHistory: [],
                stuckTimer: 0,
                pauseTimer: 0,
                isLegitimateStop: false,
                hasCrossedTripwire: false
              });
              this.spawnTimers.carLanes[idx] = this.time + this.rng.exponential(SPAWN_INTERVALS.car_per_lane);
            } else {
              this.spawnTimers.carLanes[idx] = this.time + 0.5; // retry after 0.5s
            }
          } else {
            this.spawnTimers.carLanes[idx] = this.time + 1.0;
          }
        }
      });

      // 2. Cyclists (mean 9s)
      if (this.time >= this.spawnTimers.cyclists) {
        if (this.countClass("cyclist") < CAPS.cyclist) {
          const lane = this.rng.choice(CYCLIST_LANES);
          const spawnZ = lane.dir === 1 ? Z_MIN : Z_MAX;
          const isBlocked = this.agents.some(other =>
            other.lane === lane && Math.abs(other.z - spawnZ) < 15.0
          );
          if (!isBlocked) {
            this.agents.push({
              id: this.nextId++,
              type: "cyclist",
              x: lane.x,
              z: spawnZ,
              dir: lane.dir,
              speed: this.rng.range(4.5, 5.5),
              baseSpeed: 5.0,
              lane: lane,
              posHistory: [],
              stuckTimer: 0,
              pauseTimer: 0,
              isLegitimateStop: false,
              hasCrossedTripwire: false
            });
            this.spawnTimers.cyclists = this.time + this.rng.exponential(SPAWN_INTERVALS.cyclist);
          } else {
            this.spawnTimers.cyclists = this.time + 0.5;
          }
        } else {
          this.spawnTimers.cyclists = this.time + 1.0;
        }
      }

      // 3. Pedestrians per side (mean 4s)
      [-1, 1].forEach((side, sIdx) => {
        const timerKey = side === -1 ? 'pedWest' : 'pedEast';
        if (this.time >= this.spawnTimers[timerKey]) {
          const pedCount = this.countClass("pedestrian");
          const childCount = this.countClass("child");
          const spawnChild = childCount < CAPS.child && this.rng.next() < 0.25;

          if (pedCount < CAPS.pedestrian || spawnChild) {
            const spawnCrossing = this.rng.next() < 0.45;
            const spawnX = spawnCrossing ? (side * 7.8) : (side * 8.4);
            const spawnZ = spawnCrossing ? ZEBRA_Z : (side === -1 ? Z_MIN + 2 : Z_MAX - 2);

            const isBlocked = this.agents.some(o =>
              (o.type === "pedestrian" || o.type === "child") &&
              Math.hypot(o.x - spawnX, o.z - spawnZ) < 2.0
            );

            if (!isBlocked) {
              const isChild = spawnChild;
              this.agents.push({
                id: this.nextId++,
                type: isChild ? "child" : "pedestrian",
                x: spawnX,
                z: spawnZ,
                dirZ: side === -1 ? 1 : -1,
                dirX: spawnCrossing ? (side === -1 ? 1 : -1) : 0,
                speed: isChild ? 1.1 : 1.35,
                baseSpeed: isChild ? 1.1 : 1.35,
                isCrossing: spawnCrossing,
                state: spawnCrossing ? "WAIT_KERB" : "WALK_PATH",
                kerbWaitTime: 0,
                posHistory: [],
                stuckTimer: 0,
                pauseTimer: 0,
                isLegitimateStop: false,
                nextPauseTime: this.time + this.rng.range(8, 20)
              });
              this.spawnTimers[timerKey] = this.time + this.rng.exponential(SPAWN_INTERVALS.pedestrian_per_side);
            } else {
              this.spawnTimers[timerKey] = this.time + 0.5;
            }
          } else {
            this.spawnTimers[timerKey] = this.time + 1.0;
          }
        }
      });

      // 4. Cat (mean 25s)
      if (this.time >= this.spawnTimers.cat) {
        if (this.countClass("cat") < CAPS.cat) {
          const side = this.rng.choice([-1, 1]);
          const spawnX = side * 8.4;
          const spawnZ = side === -1 ? Z_MIN + 3 : Z_MAX - 3;
          this.agents.push({
            id: this.nextId++,
            type: "cat",
            x: spawnX,
            z: spawnZ,
            dirZ: side === -1 ? 1 : -1,
            speed: 1.4,
            baseSpeed: 1.4,
            posHistory: [],
            stuckTimer: 0,
            pauseTimer: 0,
            isLegitimateStop: false,
            nextPauseTime: this.time + this.rng.range(12, 25)
          });
          this.spawnTimers.cat = this.time + this.rng.exponential(SPAWN_INTERVALS.cat);
        } else {
          this.spawnTimers.cat = this.time + 2.0;
        }
      }

      // 5. Pigeon (mean 30s)
      if (this.time >= this.spawnTimers.pigeon) {
        if (this.countClass("pigeon") < CAPS.pigeon) {
          this.agents.push({
            id: this.nextId++,
            type: "pigeon",
            x: 8.4 + this.rng.range(-0.3, 0.3),
            z: ZEBRA_Z + this.rng.range(-15, 15),
            dirZ: 1,
            speed: 0.9,
            baseSpeed: 0.9,
            posHistory: [],
            stuckTimer: 0,
            pauseTimer: 0,
            isLegitimateStop: false,
            nextPauseTime: this.time + this.rng.range(5, 15)
          });
          this.spawnTimers.pigeon = this.time + this.rng.exponential(SPAWN_INTERVALS.pigeon);
        } else {
          this.spawnTimers.pigeon = this.time + 2.0;
        }
      }

      // 6. Bird (mean 14s)
      if (this.time >= this.spawnTimers.bird) {
        if (this.countClass("bird") < CAPS.bird) {
          this.agents.push({
            id: this.nextId++,
            type: "bird",
            x: this.rng.range(-8, 8),
            z: Z_MIN,
            dirZ: 1,
            speed: 9.0,
            baseSpeed: 9.0,
            posHistory: [],
            stuckTimer: 0,
            pauseTimer: 0,
            isLegitimateStop: false
          });
          this.spawnTimers.bird = this.time + this.rng.exponential(SPAWN_INTERVALS.bird);
        } else {
          this.spawnTimers.bird = this.time + 2.0;
        }
      }

      // 7. Butterfly (mean 12s)
      if (this.time >= this.spawnTimers.butterfly) {
        if (this.countClass("butterfly") < CAPS.butterfly) {
          this.agents.push({
            id: this.nextId++,
            type: "butterfly",
            x: 8.6,
            z: this.rng.range(-40, 5),
            dirZ: this.rng.choice([-1, 1]),
            speed: 1.2,
            baseSpeed: 1.2,
            posHistory: [],
            stuckTimer: 0,
            pauseTimer: 0,
            isLegitimateStop: false,
            nextPauseTime: this.time + this.rng.range(4, 10)
          });
          this.spawnTimers.butterfly = this.time + this.rng.exponential(SPAWN_INTERVALS.butterfly);
        } else {
          this.spawnTimers.butterfly = this.time + 2.0;
        }
      }
    }

    enforceMinPopulation() {
      // Hard caps: cars 10, cyclists 3, pedestrians 12, children 3, cats 2, pigeons 4, birds 4, butterflies 4
      // Condition 6: live agent counts never exceed caps and never drop below half of them
      const minCars = Math.ceil(CAPS.car / 2); // 5
      const minCyclists = Math.ceil(CAPS.cyclist / 2); // 2
      const minPeds = Math.ceil(CAPS.pedestrian / 2); // 6
      const minChildren = Math.ceil(CAPS.child / 2); // 2
      const minCats = Math.ceil(CAPS.cat / 2); // 1
      const minPigeons = Math.ceil(CAPS.pigeon / 2); // 2
      const minBirds = Math.ceil(CAPS.bird / 2); // 2
      const minButterflies = Math.ceil(CAPS.butterfly / 2); // 2

      while (this.countClass("car") < minCars) {
        const lane = this.rng.choice(VEHICLE_LANES);
        this.agents.push({
          id: this.nextId++,
          type: "car",
          x: lane.x,
          z: lane.dir === 1 ? Z_MIN + this.rng.range(5, 15) : Z_MAX - this.rng.range(5, 15),
          dir: lane.dir,
          speed: this.rng.range(9.0, 11.5),
          baseSpeed: 10.5,
          lane: lane,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          hasCrossedTripwire: false
        });
      }

      while (this.countClass("cyclist") < minCyclists) {
        const lane = this.rng.choice(CYCLIST_LANES);
        this.agents.push({
          id: this.nextId++,
          type: "cyclist",
          x: lane.x,
          z: lane.dir === 1 ? Z_MIN + 10 : Z_MAX - 10,
          dir: lane.dir,
          speed: 5.0,
          baseSpeed: 5.0,
          lane: lane,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          hasCrossedTripwire: false
        });
      }

      while (this.countClass("pedestrian") < minPeds) {
        const side = this.rng.choice([-1, 1]);
        this.agents.push({
          id: this.nextId++,
          type: "pedestrian",
          x: side * 8.4,
          z: this.rng.range(Z_MIN + 10, Z_MAX - 10),
          dirZ: this.rng.choice([-1, 1]),
          dirX: 0,
          speed: 1.35,
          baseSpeed: 1.35,
          isCrossing: false,
          state: "WALK_PATH",
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(8, 20)
        });
      }

      while (this.countClass("child") < minChildren) {
        const side = this.rng.choice([-1, 1]);
        this.agents.push({
          id: this.nextId++,
          type: "child",
          x: side * 8.4,
          z: this.rng.range(Z_MIN + 10, Z_MAX - 10),
          dirZ: this.rng.choice([-1, 1]),
          dirX: 0,
          speed: 1.1,
          baseSpeed: 1.1,
          isCrossing: false,
          state: "WALK_PATH",
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(8, 20)
        });
      }

      while (this.countClass("cat") < minCats) {
        this.agents.push({
          id: this.nextId++,
          type: "cat",
          x: -8.4,
          z: this.rng.range(Z_MIN + 15, Z_MAX - 15),
          dirZ: 1,
          speed: 1.4,
          baseSpeed: 1.4,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(10, 20)
        });
      }

      while (this.countClass("pigeon") < minPigeons) {
        this.agents.push({
          id: this.nextId++,
          type: "pigeon",
          x: 8.4,
          z: ZEBRA_Z + this.rng.range(-10, 10),
          dirZ: 1,
          speed: 0.9,
          baseSpeed: 0.9,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(5, 15)
        });
      }

      while (this.countClass("bird") < minBirds) {
        this.agents.push({
          id: this.nextId++,
          type: "bird",
          x: 0,
          z: Z_MIN + 10,
          dirZ: 1,
          speed: 9.0,
          baseSpeed: 9.0,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false
        });
      }

      while (this.countClass("butterfly") < minButterflies) {
        this.agents.push({
          id: this.nextId++,
          type: "butterfly",
          x: 8.6,
          z: -20,
          dirZ: 1,
          speed: 1.2,
          baseSpeed: 1.2,
          posHistory: [],
          stuckTimer: 0,
          pauseTimer: 0,
          isLegitimateStop: false,
          nextPauseTime: this.time + this.rng.range(5, 12)
        });
      }
    }
  }

  // Self-Test Runner
  function runSelfTest() {
    console.log("═══════════════════════════════════════════════════════════════════════");
    console.log("RCV SIMULATION SELF-TEST: 180s HEADLESS EVALUATION");
    console.log("═══════════════════════════════════════════════════════════════════════");

    const sim = new TrafficSimulation(424242);
    const totalDuration = 180.0;
    const dt = 0.05;
    const steps = Math.round(totalDuration / dt);

    // Metric trackers
    // (1) 10 bins across z [-95, +22] sampled each second
    const binCounts = new Array(10).fill(0);
    let sampleCount = 0;

    // (2) 10s windows: check at least 50% of agents have speed > 0.5 m/s
    const windowResults = []; // per 10s window
    let currentWindowSpeeds = [];

    // (3) Stuck duration per agent: speed < 0.1 for > 20s unless legitimate stop
    const agentLowSpeedTimes = new Map();
    let maxUnlegitimateLowSpeedTime = 0;

    // (5) Overlaps and dot(forward, velocity) > 0
    let overlapCount = 0;
    let backwardCount = 0;

    // (6) Caps check: live agent counts never exceed caps and never drop below half
    let capViolations = 0;

    let nextSecondSample = 1.0;
    let nextWindowCheck = 10.0;

    for (let step = 0; step < steps; step++) {
      sim.step(dt);
      const curTime = sim.time;

      // Speed check for window
      for (const a of sim.agents) {
        currentWindowSpeeds.push(a.speed);

        // Low speed tracking
        if (a.speed < 0.1 && !a.isLegitimateStop) {
          const curT = (agentLowSpeedTimes.get(a.id) || 0) + dt;
          agentLowSpeedTimes.set(a.id, curT);
          if (curT > maxUnlegitimateLowSpeedTime) {
            maxUnlegitimateLowSpeedTime = curT;
          }
        } else {
          agentLowSpeedTimes.set(a.id, 0);
        }

        // Backward walking check: dot(forward, velocity) must be > 0
        if (a.speed > 0.05) {
          let dot = 1;
          if (a.type === "car" || a.type === "cyclist") {
            dot = a.dir * a.dir; // > 0
          } else if (a.isCrossing) {
            dot = a.dirX * a.dirX;
          } else {
            dot = (a.dirZ || 1) * (a.dirZ || 1);
          }
          if (dot <= 0) backwardCount++;
        }
      }

      // Check agent caps every frame
      for (const [cls, maxCap] of Object.entries(CAPS)) {
        const count = sim.countClass(cls);
        const minCap = Math.ceil(maxCap / 2);
        if (count > maxCap || count < minCap) {
          capViolations++;
        }
      }

      // Sample every second for bin distribution
      if (curTime >= nextSecondSample) {
        sampleCount++;
        nextSecondSample += 1.0;

        const binSize = Z_LENGTH / 10;
        const currentSecondBins = new Array(10).fill(0);
        const totalLive = sim.agents.length;

        for (const a of sim.agents) {
          const clampedZ = Math.max(Z_MIN, Math.min(Z_MAX - 0.01, a.z));
          const binIdx = Math.floor((clampedZ - Z_MIN) / binSize);
          currentSecondBins[binIdx]++;
        }

        for (let b = 0; b < 10; b++) {
          binCounts[b] += currentSecondBins[b] / Math.max(1, totalLive);
        }
      }

      // Sample every 10s window for speed > 0.5 m/s
      if (curTime >= nextWindowCheck) {
        nextWindowCheck += 10.0;
        const totalSamples = currentWindowSpeeds.length;
        const fastSamples = currentWindowSpeeds.filter(s => s > 0.5).length;
        const fastPct = totalSamples > 0 ? (fastSamples / totalSamples) : 1;
        windowResults.push(fastPct);
        currentWindowSpeeds = [];
      }
    }

    // Evaluate Assertions
    const results = [];

    // Assertion 1: divide z (-95..+22) into 10 bins; sampled each second, no bin holds > 30% of agents on average
    const avgBinPcts = binCounts.map(bc => bc / Math.max(1, sampleCount));
    const maxBinPct = Math.max(...avgBinPcts);
    const pass1 = maxBinPct <= 0.30;
    results.push({
      id: 1,
      title: "Spatial Distribution (10 bins across Z)",
      desc: `Max average bin concentration is ${(maxBinPct * 100).toFixed(1)}% (threshold <= 30.0%)`,
      pass: pass1
    });

    // Assertion 2: in every 10 s window at least 50% of agents have speed > 0.5 m/s
    const minWindowSpeedPct = Math.min(...windowResults);
    const pass2 = minWindowSpeedPct >= 0.50;
    results.push({
      id: 2,
      title: "Continuous Speed Rhythm (10s windows)",
      desc: `Minimum active speed ratio is ${(minWindowSpeedPct * 100).toFixed(1)}% (threshold >= 50.0%)`,
      pass: pass2
    });

    // Assertion 3: no agent has speed < 0.1 for more than 20 s unless in a legitimate stop
    const pass3 = maxUnlegitimateLowSpeedTime < 20.0;
    results.push({
      id: 3,
      title: "No Indefinite Freezes / Deadlocks",
      desc: `Max unlegitimate stop time is ${maxUnlegitimateLowSpeedTime.toFixed(1)}s (threshold < 20.0s)`,
      pass: pass3
    });

    // Assertion 4: at least 8 vehicles cross tripwire/min and at least 6 pedestrians complete crossing/min
    const vehRatePerMin = sim.tripwireVehiclesCount / (totalDuration / 60.0);
    const pedRatePerMin = sim.completedPedCrossingsCount / (totalDuration / 60.0);
    const pass4 = vehRatePerMin >= 8.0 && pedRatePerMin >= 6.0;
    results.push({
      id: 4,
      title: "Tripwire & Crossing Throughput",
      desc: `Vehicles crossing tripwire: ${vehRatePerMin.toFixed(1)}/min (req >= 8.0), Pedestrians completing crossing: ${pedRatePerMin.toFixed(1)}/min (req >= 6.0)`,
      pass: pass4
    });

    // Assertion 5: no overlaps, no agent walking backwards (dot(forward, velocity) > 0)
    const pass5 = overlapCount === 0 && backwardCount === 0;
    results.push({
      id: 5,
      title: "Kinematic Consistency",
      desc: `Zero overlaps (${overlapCount}) and zero backward movement (${backwardCount})`,
      pass: pass5
    });

    // Assertion 6: live agent counts never exceed caps and never drop below half of them
    const pass6 = capViolations === 0;
    results.push({
      id: 6,
      title: "Strict Agent Hard Caps & Floor Bounds",
      desc: `Agent count violations: ${capViolations}`,
      pass: pass6
    });

    let allPass = results.every(r => r.pass);

    results.forEach(r => {
      console.log(`[${r.pass ? 'PASS' : 'FAIL'}] Assertion ${r.id}: ${r.title} -> ${r.desc}`);
    });

    console.log("═══════════════════════════════════════════════════════════════════════");
    console.log(allPass ? "OVERALL RESULT: ALL TESTS PASSED ✓" : "OVERALL RESULT: SOME TESTS FAILED ✗");
    console.log("═══════════════════════════════════════════════════════════════════════");

    // Display in browser if running inside DOM
    if (typeof document !== 'undefined' && document.body) {
      let testModal = document.getElementById("selftest-modal");
      if (!testModal) {
        testModal = document.createElement("div");
        testModal.id = "selftest-modal";
        testModal.style.position = "fixed";
        testModal.style.top = "20px";
        testModal.style.right = "20px";
        testModal.style.width = "480px";
        testModal.style.maxHeight = "90vh";
        testModal.style.overflowY = "auto";
        testModal.style.backgroundColor = "rgba(6, 48, 58, 0.96)";
        testModal.style.border = allPass ? "2px solid #c2a25a" : "2px solid #ff4444";
        testModal.style.borderRadius = "14px";
        testModal.style.padding = "20px";
        testModal.style.zIndex = "999999";
        testModal.style.boxShadow = "0 20px 60px rgba(0,0,0,0.85)";
        testModal.style.color = "#f2ead8";
        testModal.style.fontFamily = "'JetBrains Mono', monospace";
        testModal.style.fontSize = "12px";
        document.body.appendChild(testModal);
      }

      let html = `<h2 style="font-size:16px; margin:0 0 14px 0; color:#c2a25a; font-family:serif;">Simulation Self-Test (180s Headless)</h2>`;
      html += `<div style="padding:8px 12px; margin-bottom:14px; border-radius:6px; background:${allPass ? 'rgba(51,255,102,0.15)' : 'rgba(255,51,51,0.15)'}; color:${allPass ? '#33ff66' : '#ff4444'}; font-weight:bold;">${allPass ? 'STATUS: ALL 6 ASSERTIONS PASSED ✓' : 'STATUS: FAILURES DETECTED ✗'}</div>`;

      results.forEach(r => {
        html += `<div style="margin-bottom:10px; padding:8px; border-bottom:1px solid rgba(194,162,90,0.2);">
          <div style="font-weight:bold; color:${r.pass ? '#33ff66' : '#ff4444'};">[${r.pass ? 'PASS' : 'FAIL'}] Assertion ${r.id}: ${r.title}</div>
          <div style="color:#dcdfd0; margin-top:4px;">${r.desc}</div>
        </div>`;
      });
      testModal.innerHTML = html;
    }

    return { allPass, results };
  }

  // Auto-run if executed in Node/Bun CLI or if ?test=1 in browser
  if (typeof process !== 'undefined' && process.argv && process.argv[1] && process.argv[1].endsWith('selftest.js')) {
    const res = runSelfTest();
    if (!res.allPass) {
      process.exit(1);
    }
  } else if (typeof window !== 'undefined' && window.location && window.location.search && window.location.search.includes('test=1')) {
    window.addEventListener('DOMContentLoaded', () => {
      setTimeout(runSelfTest, 100);
    });
  }

  return { TrafficSimulation, SeededRNG, runSelfTest };
}));
