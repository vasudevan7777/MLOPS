const form = document.getElementById("predictionForm");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const filename = document.getElementById("filename");
const progressBar = document.getElementById("progressBar");
const uploadButton = document.getElementById("uploadButton");
const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");
const preview = document.getElementById("preview");
const loader = document.getElementById("loader");
const toast = document.getElementById("toast");

const passengerCount = document.getElementById("passengerCount");
const crowdLevel = document.getElementById("crowdLevel");
const confidence = document.getElementById("confidence");
const processingTime = document.getElementById("processingTime");

function setSelectedFile(file) {
  filename.textContent = file ? file.name : "No file selected";
  progressBar.style.width = file ? "100%" : "0";
}

function resetResult() {
  preview.innerHTML = '<div class="empty-state"><span>Awaiting prediction</span></div>';
  passengerCount.textContent = "--";
  crowdLevel.textContent = "--";
  crowdLevel.className = "crowd-pill";
  confidence.textContent = "--";
  processingTime.textContent = "--";
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.style.borderColor = isError ? "rgba(239, 68, 68, 0.42)" : "rgba(34, 197, 94, 0.34)";
  toast.style.color = isError ? "#fecaca" : "#bbf7d0";
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3200);
}

function setLoading(isLoading) {
  loader.classList.toggle("active", isLoading);
  predictButton.disabled = isLoading;
  uploadButton.disabled = isLoading;
  clearButton.disabled = isLoading;
}

function renderMedia(result) {
  const cacheSafeUrl = `${result.media_url}?v=${Date.now()}`;
  if (result.media_type === "video") {
    preview.innerHTML = `<video src="${cacheSafeUrl}" controls playsinline></video>`;
    return;
  }
  preview.innerHTML = `<img src="${cacheSafeUrl}" alt="Predicted railway crowd">`;
}

function renderMetrics(result) {
  passengerCount.textContent = result.passenger_count;
  crowdLevel.textContent = result.crowd.label;
  crowdLevel.className = `crowd-pill ${result.crowd.status}`;
  confidence.textContent = `${result.confidence}%`;
  processingTime.textContent = `${result.processing_time}s`;
}

fileInput.addEventListener("change", () => {
  setSelectedFile(fileInput.files[0]);
});

uploadButton.addEventListener("click", () => {
  fileInput.click();
});

clearButton.addEventListener("click", () => {
  form.reset();
  setSelectedFile(null);
  resetResult();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    setSelectedFile(fileInput.files[0]);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    showToast("Please select a file first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Prediction failed.");
    }

    renderMedia(result);
    renderMetrics(result);
    showToast("Prediction Completed Successfully");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setLoading(false);
  }
});
