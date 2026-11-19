(() => {
    "use strict";

    const canvas = document.getElementById("pointCloudCanvas");
    const viewport = canvas.parentElement;
    const emptyMessage = document.getElementById("pointCloudEmpty");
    const input = document.getElementById("pointCloudInput");
    const status = document.getElementById("pointCloudStatus");
    const error = document.getElementById("pointCloudError");
    const fileInput = document.getElementById("pointCloudFile");
    const pointSizeSlider = document.getElementById("customRange");
    const meshRadios = document.querySelectorAll('input[name="meshRadio"]');
    const MAX_POINTS = 100000;
    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

    const state = {
        points: new Float32Array(),
        mesh: new Float32Array(),
        axes: new Float32Array([
            -1, -1, -1, 1, -1, -1,
            -1, -1, -1, -1, 1, -1,
            -1, -1, -1, -1, -1, 1
        ]),
        axisColors: new Float32Array([
            0.82, 0.27, 0.23, 0.82, 0.27, 0.23,
            0.18, 0.60, 0.34, 0.18, 0.60, 0.34,
            0.18, 0.44, 0.86, 0.18, 0.44, 0.86
        ]),
        yaw: -0.8,
        pitch: 0.45,
        distance: 4.6,
        zoomEnabled: true
    };

    let gl = canvas.getContext("webgl2", { antialias: true });
    let program;
    let locations;
    let positionBuffer;
    let colorBuffer;
    let meshBuffer;
    let axisPositionBuffer;
    let axisColorBuffer;
    let queued = false;

    function showError(message) {
        error.textContent = message;
        error.hidden = false;
    }

    function clearError() {
        error.textContent = "";
        error.hidden = true;
    }

    function compileShader(type, source) {
        const shader = gl.createShader(type);
        if (!shader) throw new Error("Unable to create a WebGL shader.");
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const message = gl.getShaderInfoLog(shader) || "Unknown shader compilation error.";
            gl.deleteShader(shader);
            throw new Error(message);
        }
        return shader;
    }

    function createProgram() {
        const vertexShader = compileShader(gl.VERTEX_SHADER, `#version 300 es
            layout(location = 0) in vec3 a_position;
            layout(location = 1) in vec3 a_color;
            uniform mat4 u_mvp;
            uniform float u_pointSize;
            uniform vec4 u_color;
            uniform bool u_useVertexColor;
            out vec4 v_color;
            void main() {
                gl_Position = u_mvp * vec4(a_position, 1.0);
                gl_PointSize = u_pointSize;
                v_color = u_useVertexColor ? vec4(a_color, 1.0) : u_color;
            }`);
        const fragmentShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
            precision mediump float;
            uniform bool u_round;
            in vec4 v_color;
            out vec4 outColor;
            void main() {
                if (u_round) {
                    vec2 offset = gl_PointCoord - 0.5;
                    if (dot(offset, offset) > 0.25) discard;
                }
                outColor = v_color;
            }`);
        const result = gl.createProgram();
        if (!result) throw new Error("Unable to create a WebGL program.");
        gl.attachShader(result, vertexShader);
        gl.attachShader(result, fragmentShader);
        gl.linkProgram(result);
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        if (!gl.getProgramParameter(result, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(result) || "Unknown WebGL program linking error.";
            gl.deleteProgram(result);
            throw new Error(message);
        }
        return result;
    }

    function uploadBuffer(buffer, data) {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    }

    function initializeRenderer() {
        if (!gl) {
            showError("This browser does not support WebGL 2, so the point cloud cannot be rendered.");
            return false;
        }
        try {
            program = createProgram();
            locations = {
                mvp: gl.getUniformLocation(program, "u_mvp"),
                pointSize: gl.getUniformLocation(program, "u_pointSize"),
                color: gl.getUniformLocation(program, "u_color"),
                useVertexColor: gl.getUniformLocation(program, "u_useVertexColor"),
                round: gl.getUniformLocation(program, "u_round")
            };
            positionBuffer = gl.createBuffer();
            colorBuffer = gl.createBuffer();
            meshBuffer = gl.createBuffer();
            axisPositionBuffer = gl.createBuffer();
            axisColorBuffer = gl.createBuffer();
            if (![positionBuffer, colorBuffer, meshBuffer, axisPositionBuffer, axisColorBuffer].every(Boolean)) {
                throw new Error("Unable to allocate WebGL point-cloud buffers.");
            }
            uploadBuffer(axisPositionBuffer, state.axes);
            uploadBuffer(axisColorBuffer, state.axisColors);
            return true;
        } catch (renderError) {
            showError(`Unable to initialize the WebGL viewer: ${renderError.message}`);
            gl = null;
            return false;
        }
    }

    function multiply(a, b) {
        const output = new Float32Array(16);
        for (let column = 0; column < 4; column++) {
            for (let row = 0; row < 4; row++) {
                for (let k = 0; k < 4; k++) {
                    output[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
                }
            }
        }
        return output;
    }

    function perspective(fov, aspect, near, far) {
        const f = 1 / Math.tan(fov / 2);
        const nf = 1 / (near - far);
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0
        ]);
    }

    function lookAt(eye, center, up) {
        let zx = eye[0] - center[0];
        let zy = eye[1] - center[1];
        let zz = eye[2] - center[2];
        let length = Math.hypot(zx, zy, zz);
        zx /= length;
        zy /= length;
        zz /= length;
        let xx = up[1] * zz - up[2] * zy;
        let xy = up[2] * zx - up[0] * zz;
        let xz = up[0] * zy - up[1] * zx;
        length = Math.hypot(xx, xy, xz);
        xx /= length;
        xy /= length;
        xz /= length;
        const yx = zy * xz - zz * xy;
        const yy = zz * xx - zx * xz;
        const yz = zx * xy - zy * xx;
        return new Float32Array([
            xx, yx, zx, 0,
            xy, yy, zy, 0,
            xz, yz, zz, 0,
            -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
            -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
            -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1
        ]);
    }

    function backgroundColor() {
        const channels = getComputedStyle(viewport).backgroundColor.match(/[\d.]+/g);
        if (!channels || channels.length < 3) return [0.96, 0.96, 0.96, 1];
        return [Number(channels[0]) / 255, Number(channels[1]) / 255, Number(channels[2]) / 255, 1];
    }

    function bindPositionBuffer(buffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    }

    function bindColorBuffer(buffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    }

    function draw() {
        if (!gl || !program) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width * dpr));
        const height = Math.max(1, Math.round(rect.height * dpr));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        gl.viewport(0, 0, width, height);
        gl.clearColor(...backgroundColor());
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const { yaw, pitch, distance } = state;
        const eye = [
            distance * Math.cos(pitch) * Math.cos(yaw),
            distance * Math.cos(pitch) * Math.sin(yaw),
            distance * Math.sin(pitch)
        ];
        const view = lookAt(eye, [0, 0, 0], [0, 0, 1]);
        const mvp = multiply(perspective(Math.PI / 4, width / height, 0.1, 50), view);
        gl.useProgram(program);
        gl.uniformMatrix4fv(locations.mvp, false, mvp);
        gl.uniform1f(locations.pointSize, Math.max(1, Number(pointSizeSlider.value)) * dpr);
        gl.uniform1i(locations.round, 0);
        gl.uniform1i(locations.useVertexColor, 1);
        bindPositionBuffer(axisPositionBuffer);
        bindColorBuffer(axisColorBuffer);
        gl.drawArrays(gl.LINES, 0, state.axes.length / 3);

        if (state.mesh.length) {
            gl.uniform1i(locations.useVertexColor, 0);
            gl.uniform4f(locations.color, 0.10, 0.16, 0.26, 0.55);
            gl.disableVertexAttribArray(1);
            bindPositionBuffer(meshBuffer);
            gl.drawArrays(gl.LINES, 0, state.mesh.length / 3);
            gl.enableVertexAttribArray(1);
        }
        if (state.points.length) {
            gl.uniform1i(locations.useVertexColor, 1);
            gl.uniform1i(locations.round, 1);
            bindPositionBuffer(positionBuffer);
            bindColorBuffer(colorBuffer);
            gl.drawArrays(gl.POINTS, 0, state.points.length / 3);
        }
    }

    function requestDraw() {
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => {
            queued = false;
            draw();
        });
    }

    function parseText(text) {
        const points = [];
        let skipped = 0;
        for (const line of text.split(/\r?\n/)) {
            let content = line.trim();
            if (!content || content.startsWith("#") || content.startsWith("//")) continue;
            content = content.replace(/[()[\]{}]/g, " ");
            const values = content.split(/[\s,;]+/).filter(Boolean).map(Number);
            if (values.length >= 3 && values.slice(0, 3).every(Number.isFinite)) {
                points.push(values.slice(0, 3));
            } else {
                skipped++;
            }
        }
        return { points, skipped };
    }

    function toPoints(value) {
        if (ArrayBuffer.isView(value)) {
            value = Array.from(value);
        }
        if (!Array.isArray(value)) {
            throw new Error("Point data must be an array of 3D points or a flat x, y, z vector.");
        }
        if (value.length && typeof value[0] === "number") {
            if (value.length > MAX_POINTS * 3) {
                throw new Error(`The viewer supports up to ${MAX_POINTS.toLocaleString()} points at a time.`);
            }
            if (value.length % 3 !== 0) {
                throw new Error("A flat point vector must contain x, y, z values in groups of three.");
            }
            const points = [];
            for (let i = 0; i < value.length; i += 3) points.push(value.slice(i, i + 3));
            value = points;
        }
        if (value.length > MAX_POINTS) {
            throw new Error(`The viewer supports up to ${MAX_POINTS.toLocaleString()} points at a time.`);
        }
        return value.map((point, index) => {
            const coordinates = Array.isArray(point)
                ? point.slice(0, 3).map(Number)
                : point && typeof point === "object"
                    ? [Number(point.x), Number(point.y), Number(point.z)]
                    : [];
            if (coordinates.length !== 3 || !coordinates.every(Number.isFinite)) {
                throw new Error(`Point ${index + 1} must contain three finite coordinates (x, y, z).`);
            }
            return coordinates;
        });
    }

    function makeMesh(points, kind) {
        if (kind === "none") return new Float32Array();
        const lines = [];
        const addEdge = (a, b) => lines.push(...points[a], ...points[b]);
        const groupSize = kind === "triangle" ? 3 : 4;
        for (let i = 0; i + groupSize <= points.length; i += groupSize) {
            for (let edge = 0; edge < groupSize; edge++) {
                addEdge(i + edge, i + (edge + 1) % groupSize);
            }
        }
        return new Float32Array(lines);
    }

    function currentMeshKind() {
        return document.querySelector('input[name="meshRadio"]:checked')?.value || "none";
    }

    function renderPoints(points, source, skipped = 0) {
        clearError();
        if (!points.length) {
            showError("No valid 3D points found. Provide x, y, and z coordinates for each point.");
            return;
        }
        if (points.length > MAX_POINTS) {
            showError(`The viewer supports up to ${MAX_POINTS.toLocaleString()} points at a time.`);
            return;
        }

        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        for (const point of points) {
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis], point[axis]);
                max[axis] = Math.max(max[axis], point[axis]);
            }
        }
        const ranges = max.map((value, axis) => value - min[axis]);
        const largestRange = Math.max(...ranges) || 1;
        const positions = new Float32Array(points.length * 3);
        const colors = new Float32Array(points.length * 3);
        for (let index = 0; index < points.length; index++) {
            const height = ranges[2] ? (points[index][2] - min[2]) / ranges[2] : 0.5;
            const color = colorAt(height);
            for (let axis = 0; axis < 3; axis++) {
                const normalizedRange = ranges[axis] / largestRange;
                positions[index * 3 + axis] = ranges[axis]
                    ? ((points[index][axis] - min[axis]) / ranges[axis] - 0.5) * normalizedRange * 2
                    : 0;
                colors[index * 3 + axis] = color[axis];
            }
        }

        state.points = positions;
        state.mesh = makeMesh(points.map((point) => point.map((coordinate, axis) =>
            ranges[axis] ? ((coordinate - min[axis]) / ranges[axis] - 0.5) * (ranges[axis] / largestRange) * 2 : 0
        )), currentMeshKind());
        if (gl) {
            uploadBuffer(positionBuffer, state.points);
            uploadBuffer(colorBuffer, colors);
            uploadBuffer(meshBuffer, state.mesh);
        }
        emptyMessage.hidden = true;
        status.textContent = `Loaded ${points.length.toLocaleString()} point${points.length === 1 ? "" : "s"} from ${source}.${skipped ? ` Skipped ${skipped} invalid line${skipped === 1 ? "" : "s"}.` : ""}`;
        requestDraw();
    }

    function colorAt(value) {
        const stops = [
            [0.24, 0.16, 0.60], [0.10, 0.45, 0.75], [0.05, 0.62, 0.55],
            [0.55, 0.75, 0.20], [0.95, 0.60, 0.10], [0.85, 0.20, 0.20]
        ];
        const offset = clamp(value, 0, 1) * (stops.length - 1);
        const start = Math.min(Math.floor(offset), stops.length - 2);
        const fraction = offset - start;
        return stops[start].map((channel, index) => channel + (stops[start + 1][index] - channel) * fraction);
    }

    function loadText(text, source) {
        const { points, skipped } = parseText(text);
        if (!points.length) {
            clearError();
            showError("No valid 3D points found. Enter one x, y, z point per line.");
            return;
        }
        if (points.length > MAX_POINTS) {
            showError(`The viewer supports up to ${MAX_POINTS.toLocaleString()} points at a time.`);
            return;
        }
        renderPoints(points, source, skipped);
    }

    function receiveScannerPoints(value) {
        try {
            renderPoints(toPoints(value), "scanner");
        } catch (pointError) {
            showError(`Could not load scanner points: ${pointError.message}`);
        }
    }

    function makePreset() {
        const points = [];
        const columns = 25;
        const rows = 13;
        for (let row = 0; row < rows; row++) {
            const latitude = (-75 + 150 * row / (rows - 1)) * Math.PI / 180;
            for (let column = 0; column < columns; column++) {
                const longitude = 2 * Math.PI * column / (columns - 1);
                points.push([
                    Math.cos(latitude) * Math.cos(longitude),
                    Math.cos(latitude) * Math.sin(longitude),
                    Math.sin(latitude)
                ]);
            }
        }
        return points;
    }

    function makeRandomCloud() {
        return Array.from({ length: 500 }, () => {
            const z = Math.random() * 2 - 1;
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.sqrt(1 - z * z) * (0.65 + Math.random() * 0.35);
            return [radius * Math.cos(angle), radius * Math.sin(angle), z];
        });
    }

    document.getElementById("pointCloudPlot").addEventListener("click", () => {
        loadText(input.value, "manual input");
    });
    document.getElementById("pointCloudPreset").addEventListener("click", () => {
        renderPoints(makePreset(), "preset point cloud");
    });
    document.getElementById("pointCloudRandom").addEventListener("click", () => {
        renderPoints(makeRandomCloud(), "random generator");
    });
    fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".csv")) {
            showError("Please choose a .csv file.");
            fileInput.value = "";
            return;
        }
        try {
            const text = await file.text();
            input.value = text;
            loadText(text, file.name);
        } catch (fileError) {
            showError(`Could not read ${file.name}: ${fileError.message}`);
        } finally {
            fileInput.value = "";
        }
    });

    pointSizeSlider.addEventListener("input", requestDraw);
    meshRadios.forEach((radio) => {
        radio.addEventListener("change", () => {
            if (!state.points.length) return;
            const points = [];
            for (let index = 0; index < state.points.length; index += 3) {
                points.push([state.points[index], state.points[index + 1], state.points[index + 2]]);
            }
            state.mesh = makeMesh(points, currentMeshKind());
            if (gl) uploadBuffer(meshBuffer, state.mesh);
            requestDraw();
        });
    });
    document.querySelectorAll('input[name="zoomRadio"]').forEach((radio) => {
        radio.addEventListener("change", () => {
            state.zoomEnabled = document.getElementById("zoomOn").checked;
        });
    });

    const pointers = new Map();
    let lastPinchDistance = 0;
    const pinchDistance = () => {
        const [first, second] = [...pointers.values()];
        return Math.hypot(first.x - second.x, first.y - second.y);
    };
    canvas.addEventListener("pointerdown", (event) => {
        canvas.setPointerCapture(event.pointerId);
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (pointers.size === 2) lastPinchDistance = pinchDistance();
    });
    canvas.addEventListener("pointermove", (event) => {
        const pointer = pointers.get(event.pointerId);
        if (!pointer) return;
        const dx = event.clientX - pointer.x;
        const dy = event.clientY - pointer.y;
        pointer.x = event.clientX;
        pointer.y = event.clientY;
        if (pointers.size === 1) {
            state.yaw -= dx * 0.008;
            state.pitch = clamp(state.pitch + dy * 0.008, -1.5, 1.5);
        } else if (pointers.size === 2 && state.zoomEnabled) {
            const distance = pinchDistance();
            if (lastPinchDistance > 0 && distance > 0) {
                state.distance = clamp(state.distance * lastPinchDistance / distance, 1.5, 14);
            }
            lastPinchDistance = distance;
        }
        requestDraw();
    });
    const endPointer = (event) => {
        pointers.delete(event.pointerId);
        lastPinchDistance = 0;
    };
    canvas.addEventListener("pointerup", endPointer);
    canvas.addEventListener("pointercancel", endPointer);
    canvas.addEventListener("wheel", (event) => {
        if (!state.zoomEnabled) return;
        event.preventDefault();
        state.distance = clamp(state.distance * Math.exp(event.deltaY * 0.0012), 1.5, 14);
        requestDraw();
    }, { passive: false });

    window.addEventListener("pointcloud:points", (event) => {
        receiveScannerPoints(event.detail?.points ?? event.detail);
    });
    window.updatePointCloud = receiveScannerPoints;

    if (initializeRenderer()) {
        new ResizeObserver(requestDraw).observe(viewport);
        requestDraw();
    }
})();
