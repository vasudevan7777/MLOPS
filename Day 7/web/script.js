const form = document.getElementById("predictionForm");
const modelStatus = document.getElementById("modelStatus");
const clock = document.getElementById("clock");
const predictButton = document.getElementById("predictButton");
const swapRouteButton = document.getElementById("swapRouteButton");
const formMessage = document.getElementById("formMessage");
const predictionLabel = document.getElementById("predictionLabel");
const resultCard = document.getElementById("resultCard");
const resultSummary = document.getElementById("resultSummary");
const occupancyValue = document.getElementById("occupancyValue");
const bookedRatio = document.getElementById("bookedRatio");
const seatBalance = document.getElementById("seatBalance");
const waitingLoad = document.getElementById("waitingLoad");
const probabilityList = document.getElementById("probabilityList");
const confidenceLabel = document.getElementById("confidenceLabel");
const adviceList = document.getElementById("adviceList");
const trainingRows = document.getElementById("trainingRows");
const modelAccuracy = document.getElementById("modelAccuracy");
const classSpread = document.getElementById("classSpread");
const averageOccupancy = document.getElementById("averageOccupancy");
const routeList = document.getElementById("routeList");

const controls = {
    train_capacity: document.getElementById("trainCapacity"),
    booked_seats: document.getElementById("bookedSeats"),
    available_seats: document.getElementById("availableSeats"),
    waiting_list_count: document.getElementById("waitingList"),
    day_type: document.getElementById("dayType"),
    festival_flag: document.getElementById("festivalFlag"),
    source_station: document.getElementById("sourceStation"),
    destination_station: document.getElementById("destinationStation"),
};

const fallbackOptions = {
    day_type: ["Weekday", "Weekend"],
    festival_flag: ["No", "Yes"],
    source_station: ["Chennai", "Bangalore", "Mumbai", "Delhi", "Kolkata", "Hyderabad", "Pune"],
    destination_station: ["Bangalore", "Chennai", "Delhi", "Mumbai", "Hyderabad", "Kolkata", "Pune"],
};

const levelMessages = {
    LOW: "Low crowd pressure expected. Normal boarding and coach allocation should be enough for this journey.",
    MEDIUM: "Medium crowd pressure expected. Keep platform flow monitored and prepare staff for boarding peaks.",
    HIGH: "High crowd pressure expected. Consider extra coach planning, queue control, and stronger platform supervision.",
};

const numberFormatter = new Intl.NumberFormat("en-IN");
const percentFormatter = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 });
const API_ORIGIN =
    window.location.protocol === "file:" || (window.location.port && window.location.port !== "5002")
        ? "http://127.0.0.1:5002"
        : "";

function apiUrl(path) {
    return `${API_ORIGIN}${path}`;
}

function setClock() {
    clock.textContent = new Date().toLocaleTimeString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
    });
}

function formatNumber(value) {
    return Number.isFinite(value) ? numberFormatter.format(value) : "--";
}

function formatPercent(value) {
    return Number.isFinite(value) ? `${percentFormatter.format(value)}%` : "--%";
}

function fillSelect(select, values, preferredValue) {
    select.innerHTML = "";
    values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
    });
    if (preferredValue && values.includes(preferredValue)) {
        select.value = preferredValue;
    }
}

function updateDerivedStats() {
    const capacity = Number(controls.train_capacity.value);
    const booked = Number(controls.booked_seats.value);
    const available = Number(controls.available_seats.value);
    const waiting = Number(controls.waiting_list_count.value);
    const occupancy = capacity > 0 ? (booked / capacity) * 100 : NaN;
    const waitingPercent = capacity > 0 ? (waiting / capacity) * 100 : NaN;

    bookedRatio.textContent = formatPercent(occupancy);
    occupancyValue.textContent = formatPercent(occupancy);
    seatBalance.textContent = Number.isFinite(capacity) ? formatNumber(capacity - booked - available) : "--";
    waitingLoad.textContent = formatPercent(waitingPercent);
}

function getPayload() {
    return {
        train_capacity: Number(controls.train_capacity.value),
        booked_seats: Number(controls.booked_seats.value),
        available_seats: Number(controls.available_seats.value),
        waiting_list_count: Number(controls.waiting_list_count.value),
        day_type: controls.day_type.value,
        festival_flag: controls.festival_flag.value,
        source_station: controls.source_station.value,
        destination_station: controls.destination_station.value,
    };
}

function validatePayload(payload) {
    if (payload.source_station === payload.destination_station) {
        return "Source and destination station must be different.";
    }
    if (payload.booked_seats + payload.available_seats > payload.train_capacity) {
        return "Booked seats plus available seats cannot exceed train capacity.";
    }
    return "";
}

function renderProbabilities(probabilities = {}, prediction) {
    const entries = Object.keys(probabilities).length
        ? Object.entries(probabilities)
        : [["LOW", 0], ["MEDIUM", 0], ["HIGH", 0]];

    probabilityList.innerHTML = "";
    entries.forEach(([label, rawValue]) => {
        const value = Number(rawValue) * 100;
        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
            <div class="probability-head">
                <span>${label}</span>
                <strong>${formatPercent(value)}</strong>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: ${Math.max(0, Math.min(100, value))}%"></div>
            </div>
        `;
        if (label === prediction) {
            row.querySelector(".probability-fill").style.background = "var(--teal)";
        }
        probabilityList.appendChild(row);
    });

    const best = entries.reduce(
        (winner, current) => (Number(current[1]) > Number(winner[1]) ? current : winner),
        entries[0]
    );
    confidenceLabel.textContent = `${best[0]} confidence ${formatPercent(Number(best[1]) * 100)}`;
}

function renderAdvice(level, payload) {
    const occupancy = payload.train_capacity > 0 ? (payload.booked_seats / payload.train_capacity) * 100 : 0;
    const advice = {
        LOW: [
            "Run normal coach allocation for this train.",
            "Keep standard station staff coverage active.",
            "Use available seats for last-minute passengers if the route permits.",
        ],
        MEDIUM: [
            "Monitor boarding gates during peak arrival time.",
            "Keep one backup staff team ready near the platform.",
            "Review waiting list movement before departure.",
        ],
        HIGH: [
            "Plan extra coach support or an additional service if available.",
            "Deploy queue control at platform entry and coach doors.",
            "Send early crowd alerts to station staff and passengers.",
        ],
    };

    const items = [...(advice[level] || advice.MEDIUM)];
    if (occupancy > 85) {
        items.push("Occupancy is already above 85%, so treat this journey as capacity sensitive.");
    }

    adviceList.innerHTML = "";
    items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        adviceList.appendChild(li);
    });
}

function renderPrediction(data, payload) {
    const prediction = String(data.prediction || "UNKNOWN").toUpperCase();
    predictionLabel.textContent = prediction;
    resultCard.dataset.level = prediction;
    resultSummary.textContent = levelMessages[prediction] || "Prediction completed for this train journey.";
    renderProbabilities(data.probabilities, prediction);
    renderAdvice(prediction, payload);
}

async function runPrediction(event) {
    event.preventDefault();
    formMessage.textContent = "";
    const payload = getPayload();
    const error = validatePayload(payload);
    if (error) {
        formMessage.textContent = error;
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";
    try {
        const response = await fetch(apiUrl("/api/predict"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await readJsonResponse(response);
        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }
        renderPrediction(data, payload);
        modelStatus.textContent = "Online";
    } catch (error) {
        formMessage.textContent = error.message;
        modelStatus.textContent = "Check server";
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Run prediction";
        updateDerivedStats();
        setClock();
    }
}

async function loadOptions() {
    let options = fallbackOptions;
    try {
        const response = await fetch(apiUrl("/api/options"));
        if (response.ok) {
            const data = await readJsonResponse(response);
            options = { ...fallbackOptions, ...(data.options || {}) };
        }
    } catch {
        modelStatus.textContent = "Local options";
    }

    fillSelect(controls.day_type, options.day_type, "Weekday");
    fillSelect(controls.festival_flag, options.festival_flag, "No");
    fillSelect(controls.source_station, options.source_station, "Chennai");
    fillSelect(controls.destination_station, options.destination_station, "Bangalore");
}

async function loadModelInfo() {
    try {
        const response = await fetch(apiUrl("/api/model-info"));
        if (!response.ok) {
            throw new Error("Model info unavailable");
        }
        const data = await readJsonResponse(response);
        trainingRows.textContent = formatNumber(Number(data.rows_used));
        modelAccuracy.textContent = formatPercent(Number(data.accuracy) * 100);
        const counts = data.target_counts || {};
        classSpread.textContent = `${formatNumber(counts.low || 0)} / ${formatNumber(counts.medium || 0)} / ${formatNumber(counts.high || 0)}`;
        modelStatus.textContent = "Online";
    } catch {
        trainingRows.textContent = "--";
        modelAccuracy.textContent = "--";
        classSpread.textContent = "--";
    }
}

async function loadAnalytics() {
    try {
        const response = await fetch(apiUrl("/api/analytics"));
        if (!response.ok) {
            throw new Error("Analytics unavailable");
        }
        const data = await readJsonResponse(response);
        const summary = data.dataset_summary || {};
        averageOccupancy.textContent = formatPercent(Number(summary.avg_occupancy));
        renderRoutes((data.route_stats || {}).top_routes || []);
    } catch {
        averageOccupancy.textContent = "--";
        renderRoutes([]);
    }
}

async function readJsonResponse(response) {
    const text = await response.text();
    if (!text.trim()) {
        throw new Error(
            "Empty server response. Start Flask with `python app.py` and open http://127.0.0.1:5002/."
        );
    }
    try {
        return JSON.parse(text);
    } catch (error) {
        throw new Error(`Server returned invalid JSON from ${response.url}.`);
    }
}

function renderRoutes(routes) {
    const items = routes.length
        ? routes
        : [
              { route: "Chennai -> Bangalore", count: 0 },
              { route: "Mumbai -> Pune", count: 0 },
              { route: "Delhi -> Jaipur", count: 0 },
          ];

    routeList.innerHTML = "";
    items.slice(0, 5).forEach((route) => {
        const card = document.createElement("div");
        card.className = "route-item";
        card.innerHTML = `
            <strong>${route.route}</strong>
            <span>${formatNumber(Number(route.count))} records</span>
        `;
        routeList.appendChild(card);
    });
}

function swapRoute() {
    const source = controls.source_station.value;
    controls.source_station.value = controls.destination_station.value;
    controls.destination_station.value = source;
}

async function init() {
    setClock();
    renderProbabilities();
    await loadOptions();
    updateDerivedStats();
    loadModelInfo();
    loadAnalytics();
}

Object.values(controls).forEach((control) => {
    control.addEventListener("input", updateDerivedStats);
});

form.addEventListener("submit", runPrediction);
swapRouteButton.addEventListener("click", swapRoute);
setInterval(setClock, 60000);
init();
