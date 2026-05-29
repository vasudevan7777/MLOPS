const form = document.getElementById("predictForm");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");
const button = document.getElementById("predictBtn");
const resultBadge = document.getElementById("resultBadge");
const probabilitiesEl = document.getElementById("probabilities");
const modelStatus = document.getElementById("modelStatus");
const datasetRows = document.getElementById("datasetRows");
const modelAccuracy = document.getElementById("modelAccuracy");
const lastPrediction = document.getElementById("lastPrediction");
const labelSummary = document.getElementById("labelSummary");
const sourceOptions = document.getElementById("sourceOptions");
const destinationOptions = document.getElementById("destinationOptions");
const swapRouteBtn = document.getElementById("swapRouteBtn");
const routeSummary = document.getElementById("routeSummary");

const FLASK_BASE_URL = "http://127.0.0.1:5002";
const isLiveServer = window.location.port === "5500" || window.location.protocol === "file:";
const API_BASE = isLiveServer ? FLASK_BASE_URL : "";
let useLocalModel = false;
let localModelData = null;

const fields = {
    age: document.getElementById("age"),
    gender: document.getElementById("gender"),
    booking_type: document.getElementById("bookingType"),
    source: document.getElementById("source"),
    destination: document.getElementById("destination"),
    fare: document.getElementById("fare"),
    ticket_status: document.getElementById("ticketStatus"),
};

const levelClass = {
    LOW: "low",
    MEDIUM: "medium",
    HIGH: "high",
};

function fillOptions(select, values) {
    select.innerHTML = "";
    values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
    });
}

function fillRouteInputs(sourceValues, destinationValues) {
    fillOptions(sourceOptions, sourceValues);
    fillOptions(destinationOptions, destinationValues);
    fields.source.value = sourceValues.includes("Chennai") ? "Chennai" : sourceValues[0] || "";
    fields.destination.value = destinationValues.includes("Bangalore") ? "Bangalore" : destinationValues[1] || "";
    updateRouteSummary();
}

function setResult(label, isError = false) {
    resultEl.textContent = label;
    resultEl.className = "result-text";
    resultBadge.className = "tag neutral";
    errorEl.textContent = "";

    const cssClass = levelClass[label];
    if (cssClass) {
        resultEl.classList.add(cssClass);
        resultBadge.className = `tag ${cssClass}`;
        resultBadge.textContent = label;
        return;
    }

    resultBadge.textContent = isError ? "Error" : "Ready";
}

function renderProbabilities(probabilities) {
    probabilitiesEl.innerHTML = "";
    Object.entries(probabilities || {}).forEach(([label, value]) => {
        const row = document.createElement("div");
        const percentage = Math.round(value * 100);
        row.className = "probability";
        row.innerHTML = `
            <div class="probability-top">
                <span>${label}</span>
                <strong>${percentage}%</strong>
            </div>
            <div class="bar"><span style="width: ${percentage}%"></span></div>
        `;
        probabilitiesEl.appendChild(row);
    });
}

function normalizeInfo(data) {
    return data.metrics ? data.metrics : data;
}

function renderModelInfo(infoSource) {
    const info = normalizeInfo(infoSource);
    datasetRows.textContent = info.rows_used ?? "--";
    modelAccuracy.textContent =
        info.accuracy === null || info.accuracy === undefined
            ? "--"
            : `${Math.round(info.accuracy * 100)}%`;

    labelSummary.innerHTML = "";
    Object.entries(info.target_counts || {}).forEach(([label, count]) => {
        const range = info.fare_ranges?.[label];
        const item = document.createElement("div");
        item.className = `label-card ${levelClass[label.toUpperCase()] || ""}`;
        item.innerHTML = `
            <span>${label.toUpperCase()}</span>
            <strong>${count}</strong>
            <small>${range ? `${range.min} - ${range.max}` : "--"}</small>
        `;
        labelSummary.appendChild(item);
    });
}

async function getJson(url, options = undefined) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || `${url} failed.`);
    }
    return data;
}

async function loadLocalModelData() {
    localModelData = await getJson("model_data.json");
    useLocalModel = true;
    modelStatus.textContent = "Model ready (Go Live)";

    const options = localModelData.options;
    fillOptions(fields.gender, options.gender || []);
    fillOptions(fields.booking_type, options.booking_type || []);
    fillRouteInputs(options.source || [], options.destination || []);
    fillOptions(fields.ticket_status, options.ticket_status || []);
    renderModelInfo(localModelData);
}

async function loadFlaskModelData() {
    const [optionsData, infoData] = await Promise.all([
        getJson(`${API_BASE}/api/options`),
        getJson(`${API_BASE}/api/model-info`),
    ]);

    const options = optionsData.options;
    fillOptions(fields.gender, options.gender || []);
    fillOptions(fields.booking_type, options.booking_type || []);
    fillRouteInputs(options.source || [], options.destination || []);
    fillOptions(fields.ticket_status, options.ticket_status || []);
    renderModelInfo(infoData);
    modelStatus.textContent = "Model ready";
}

function getPayload() {
    return Object.fromEntries(
        Object.entries(fields).map(([name, field]) => [name, field.value])
    );
}

function validatePayload(payload) {
    if (payload.source.trim().toLowerCase() === payload.destination.trim().toLowerCase()) {
        throw new Error("Source and destination must be different.");
    }

    if (!Number.isFinite(Number(payload.fare)) || Number(payload.fare) <= 0) {
        throw new Error("Fare must be greater than zero.");
    }
}

function updateRouteSummary(prediction = null) {
    const source = fields.source.value || "Source";
    const destination = fields.destination.value || "Destination";
    const fare = fields.fare.value || "--";
    const suffix = prediction ? ` | ${prediction}` : "";
    routeSummary.textContent = `${source} to ${destination} | Fare ${fare}${suffix}`;
}

function predictWithLocalModel(payload) {
    const fare = Number(payload.fare);
    const ranges = localModelData.metrics.fare_ranges;
    const labels = ["LOW", "MEDIUM", "HIGH"];
    const match = labels.find((label) => {
        const range = ranges[label.toLowerCase()];
        return range && fare >= range.min && fare <= range.max;
    });
    const prediction = match || (fare < ranges.low.min ? "LOW" : "HIGH");
    const probabilities = Object.fromEntries(labels.map((label) => [label, label === prediction ? 1 : 0]));
    return { prediction, probabilities };
}

async function predictWithFlask(payload) {
    return getJson(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    button.disabled = true;
    button.textContent = "Predicting";
    setResult("...");
    renderProbabilities({});

    try {
        const payload = getPayload();
        validatePayload(payload);
        const data = useLocalModel ? predictWithLocalModel(payload) : await predictWithFlask(payload);
        setResult(data.prediction);
        updateRouteSummary(data.prediction);
        lastPrediction.textContent = data.prediction;
        renderProbabilities(data.probabilities);
    } catch (error) {
        setResult("--", true);
        errorEl.textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Predict Crowd";
    }
});

swapRouteBtn.addEventListener("click", () => {
    const source = fields.source.value;
    fields.source.value = fields.destination.value;
    fields.destination.value = source;
    updateRouteSummary();
});

["source", "destination", "fare"].forEach((name) => {
    fields[name].addEventListener("input", () => updateRouteSummary());
});

(async function init() {
    try {
        await loadFlaskModelData();
    } catch (error) {
        try {
            await loadLocalModelData();
        } catch (fallbackError) {
            modelStatus.textContent = "Model error";
            errorEl.textContent = fallbackError.message;
        }
    }
})();
