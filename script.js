/* ==========================================================================
   script.js
   Handles: form validation, calling the FastAPI backend with fetch(),
   showing loading states, animated number reveal, and rendering the
   prediction / what-if / feature importance results into the page.
   ========================================================================== */

const form = document.getElementById("employee-form");
const predictBtn = document.getElementById("predict-btn");
const formError = document.getElementById("form-error");

const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const resultsEl = document.getElementById("results");

const whatIfBtn = document.getElementById("whatif-btn");
const whatIfResults = document.getElementById("whatif-results");

const downloadReportBtn = document.getElementById("download-report-btn");
const downloadReportLabel = document.getElementById("download-report-label");
const reportError = document.getElementById("report-error");

const saveProfileBtn = document.getElementById("save-profile-btn");
const saveProfileLabel = document.getElementById("save-profile-label");
const saveSuccess = document.getElementById("save-success");

const savedEmpty = document.getElementById("saved-empty");
const savedList = document.getElementById("saved-list");
const compareBtn = document.getElementById("compare-btn");
const compareTableWrap = document.getElementById("compare-table-wrap");
const compareTable = document.getElementById("compare-table");

// Keeps the last submitted, validated employee data so the
// What-If simulator can reuse it without re-reading the form.
let lastEmployeeData = null;
let lastPredictionResult = null;

const SAVED_PROFILES_KEY = "salaryApp.savedProfiles";
const MAX_SAVED_PROFILES = 12;

/* -------------------------------------------------------------------- */
/*  Helper: read and validate the form                                   */
/* -------------------------------------------------------------------- */

function clearFieldErrors() {
    document.querySelectorAll(".field-error").forEach(el => (el.textContent = ""));
    formError.textContent = "";
}

function readEmployeeDataFromForm() {
    return {
        age: Number(document.getElementById("age").value),
        education_level: document.getElementById("education_level").value,
        occupation: document.getElementById("occupation").value,
        years_of_experience: Number(document.getElementById("years_of_experience").value),
        hours_per_week: Number(document.getElementById("hours_per_week").value),
        gender: document.getElementById("gender").value,
        country: document.getElementById("country").value,
        industry: document.getElementById("industry").value,
    };
}

/**
 * Simple client-side validation, mirroring the rules enforced server-side
 * by Pydantic. This gives the user instant feedback before we even call
 * the API.
 */
function validateEmployeeData(data) {
    let isValid = true;

    if (!Number.isFinite(data.age) || data.age < 18 || data.age > 70) {
        document.getElementById("error-age").textContent = "Age must be between 18 and 70.";
        isValid = false;
    }
    if (!Number.isFinite(data.years_of_experience) || data.years_of_experience < 0 || data.years_of_experience > 50) {
        document.getElementById("error-years_of_experience").textContent = "Experience must be between 0 and 50 years.";
        isValid = false;
    }
    if (!Number.isFinite(data.hours_per_week) || data.hours_per_week < 1 || data.hours_per_week > 100) {
        document.getElementById("error-hours_per_week").textContent = "Hours must be between 1 and 100.";
        isValid = false;
    }

    return isValid;
}

/* -------------------------------------------------------------------- */
/*  Helper: formatting                                                   */
/* -------------------------------------------------------------------- */

function formatCurrency(value) {
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
    }).format(value);
}

/**
 * Animates a number counting up from 0 to `target` inside `element`,
 * formatted as currency. Uses requestAnimationFrame with an ease-out
 * curve so the motion feels quick at first and settles smoothly.
 */
function animateNumberTo(element, target, duration = 900) {
    const start = performance.now();

    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(target * eased);
        element.textContent = formatCurrency(current);
        if (progress < 1) {
            requestAnimationFrame(tick);
        } else {
            element.textContent = formatCurrency(target);
        }
    }

    requestAnimationFrame(tick);
}

/* -------------------------------------------------------------------- */
/*  Handle "Predict my salary"                                           */
/* -------------------------------------------------------------------- */

form.addEventListener("submit", async function (event) {
    event.preventDefault();
    clearFieldErrors();

    const employeeData = readEmployeeDataFromForm();

    if (!validateEmployeeData(employeeData)) {
        formError.textContent = "Please fix the highlighted fields above.";
        return;
    }

    lastEmployeeData = employeeData;

    // Show loading state
    emptyState.classList.add("hidden");
    resultsEl.classList.add("hidden");
    whatIfResults.classList.add("hidden");
    loadingState.classList.remove("hidden");
    predictBtn.disabled = true;
    predictBtn.querySelector("span").textContent = "Predicting…";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(employeeData),
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => null);
            const message = errorBody?.detail
                ? typeof errorBody.detail === "string"
                    ? errorBody.detail
                    : "Please check your input values and try again."
                : "Something went wrong while predicting your salary.";
            throw new Error(message);
        }

        const result = await response.json();
        renderPredictionResult(result);
        await loadFeatureImportance(employeeData);
    } catch (error) {
        formError.textContent = error.message || "Unable to reach the server. Is the backend running?";
        emptyState.classList.remove("hidden");
    } finally {
        loadingState.classList.add("hidden");
        predictBtn.disabled = false;
        predictBtn.querySelector("span").textContent = "Predict my salary";
    }
});

function renderPredictionResult(result) {
    const salaryEl = document.getElementById("predicted-salary");
    salaryEl.textContent = formatCurrency(0);

    document.getElementById("salary-range").textContent =
        `Estimated range: ${formatCurrency(result.salary_range.minimum)} – ${formatCurrency(result.salary_range.maximum)}`;
    document.getElementById("experience-level").textContent = result.experience_level;
    document.getElementById("salary-position").textContent = result.salary_insight;
    document.getElementById("career-insight-text").textContent = result.career_insight;

    resultsEl.classList.remove("hidden");

    // Animate the headline number counting up once the card is visible
    animateNumberTo(salaryEl, result.predicted_salary);

    // A fresh prediction always starts as "not yet saved" for this profile
    lastPredictionResult = result;
    saveProfileBtn.classList.remove("saved");
    saveProfileLabel.textContent = "Save this profile";
    saveSuccess.textContent = "";
}

/* -------------------------------------------------------------------- */
/*  Handle "Analyze salary growth" (What-If simulator)                   */
/* -------------------------------------------------------------------- */

whatIfBtn.addEventListener("click", async function () {
    if (!lastEmployeeData) {
        formError.textContent = "Please predict your salary first.";
        return;
    }

    const experienceChange = Number(document.getElementById("whatif-experience").value) || 0;
    const hoursChange = Number(document.getElementById("whatif-hours").value) || 0;

    whatIfBtn.disabled = true;
    whatIfBtn.querySelector("span").textContent = "Analyzing…";

    try {
        const response = await fetch("/what-if", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                employee: lastEmployeeData,
                experience_change: experienceChange,
                hours_change: hoursChange,
            }),
        });

        if (!response.ok) {
            throw new Error("Unable to run the what-if simulation.");
        }

        const result = await response.json();

        whatIfResults.classList.add("hidden");

        document.getElementById("whatif-current").textContent = formatCurrency(result.current_salary);
        document.getElementById("whatif-new").textContent = formatCurrency(result.new_salary);
        document.getElementById("whatif-diff").textContent =
            (result.difference >= 0 ? "+" : "") + formatCurrency(result.difference);
        document.getElementById("whatif-explanation").textContent = result.explanation;

        // Re-trigger the reveal animation
        void whatIfResults.offsetWidth;
        whatIfResults.classList.remove("hidden");
    } catch (error) {
        formError.textContent = error.message;
    } finally {
        whatIfBtn.disabled = false;
        whatIfBtn.querySelector("span").textContent = "Analyze salary growth";
    }
});

/* -------------------------------------------------------------------- */
/*  Handle "Download PDF report"                                          */
/* -------------------------------------------------------------------- */

downloadReportBtn.addEventListener("click", async function () {
    reportError.textContent = "";

    if (!lastEmployeeData) {
        reportError.textContent = "Please predict your salary first.";
        return;
    }

    downloadReportBtn.disabled = true;
    downloadReportLabel.textContent = "Preparing report…";

    try {
        const response = await fetch("/generate-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(lastEmployeeData),
        });

        if (!response.ok) {
            throw new Error("Unable to generate the PDF report.");
        }

        // The backend streams back raw PDF bytes. We turn that into a Blob,
        // create a temporary object URL for it, and click a hidden link to
        // trigger the browser's normal "Save File" download flow.
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);

        const tempLink = document.createElement("a");
        tempLink.href = downloadUrl;
        tempLink.download = "salary_prediction_report.pdf";
        document.body.appendChild(tempLink);
        tempLink.click();
        document.body.removeChild(tempLink);

        window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
        reportError.textContent = error.message || "Something went wrong while generating the report.";
    } finally {
        downloadReportBtn.disabled = false;
        downloadReportLabel.textContent = "Download PDF report";
    }
});

/* -------------------------------------------------------------------- */
/*  Feature importance (personalized to the current profile)             */
/* -------------------------------------------------------------------- */

async function loadFeatureImportance(employeeData) {
    const container = document.getElementById("importance-list");
    container.innerHTML = "";

    try {
        const response = await fetch("/feature-importance", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(employeeData),
        });
        const data = await response.json();

        if (!data.supported || !data.top_factors || data.top_factors.length === 0) {
            container.innerHTML = `<p class="panel-hint">Could not compute factors for this profile.</p>`;
            return;
        }

        const maxAbsImpact = Math.max(...data.top_factors.map(f => Math.abs(f.impact)), 1);
        const bars = [];

        data.top_factors.forEach(factor => {
            const percentage = Math.round((Math.abs(factor.impact) / maxAbsImpact) * 100);
            const isPositive = factor.direction === "positive";
            const sign = isPositive ? "+" : "−";
            const impactLabel = `${sign}${formatCurrency(Math.abs(factor.impact))}`;

            const row = document.createElement("div");
            row.className = `importance-item ${isPositive ? "positive" : "negative"}`;
            row.innerHTML = `
                <span class="importance-feature-name">
                    ${formatFeatureName(factor.feature)}
                    <span class="importance-your-value">${factor.your_value}</span>
                </span>
                <div class="importance-bar-track">
                    <div class="importance-bar-fill" data-width="${percentage}"></div>
                </div>
                <span class="importance-value">${impactLabel}</span>
            `;
            container.appendChild(row);
            bars.push(row.querySelector(".importance-bar-fill"));
        });

        // Animate bars filling in on a short stagger, after layout settles
        requestAnimationFrame(() => {
            bars.forEach((bar, index) => {
                setTimeout(() => {
                    bar.style.width = bar.dataset.width + "%";
                }, index * 80);
            });
        });
    } catch (error) {
        container.innerHTML = `<p class="panel-hint">Could not load feature importance.</p>`;
    }
}

function formatFeatureName(name) {
    return name
        .split("_")
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

/* -------------------------------------------------------------------- */
/*  Saved profiles (local storage) + compare                             */
/* -------------------------------------------------------------------- */

function loadSavedProfiles() {
    try {
        const raw = localStorage.getItem(SAVED_PROFILES_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (error) {
        console.error("Could not read saved profiles:", error);
        return [];
    }
}

function persistSavedProfiles(profiles) {
    try {
        localStorage.setItem(SAVED_PROFILES_KEY, JSON.stringify(profiles));
    } catch (error) {
        console.error("Could not save profiles:", error);
    }
}

function renderSavedProfiles() {
    const profiles = loadSavedProfiles();

    if (profiles.length === 0) {
        savedEmpty.classList.remove("hidden");
        savedList.innerHTML = "";
        compareBtn.disabled = true;
        compareTableWrap.classList.add("hidden");
        return;
    }

    savedEmpty.classList.add("hidden");
    savedList.innerHTML = "";

    profiles.forEach(profile => {
        const card = document.createElement("div");
        card.className = "saved-card" + (profile.selected ? " selected" : "");
        card.innerHTML = `
            <div class="saved-card-top">
                <span class="saved-card-name">${escapeHtml(profile.name)}</span>
            </div>
            <div class="saved-card-salary">${formatCurrency(profile.prediction.predicted_salary)}</div>
            <div class="saved-card-meta">
                ${escapeHtml(profile.employee.occupation)} · ${escapeHtml(profile.employee.country)}<br>
                ${profile.employee.years_of_experience} yrs exp · ${escapeHtml(profile.employee.education_level)}
            </div>
            <div class="saved-card-actions">
                <label class="saved-card-checkbox">
                    <input type="checkbox" data-id="${profile.id}" class="compare-checkbox" ${profile.selected ? "checked" : ""}>
                    Compare
                </label>
                <button type="button" class="saved-card-remove" data-id="${profile.id}">Remove</button>
            </div>
        `;
        savedList.appendChild(card);
    });

    savedList.querySelectorAll(".compare-checkbox").forEach(box => {
        box.addEventListener("change", function () {
            toggleProfileSelected(this.dataset.id, this.checked);
        });
    });

    savedList.querySelectorAll(".saved-card-remove").forEach(btn => {
        btn.addEventListener("click", function () {
            removeSavedProfile(this.dataset.id);
        });
    });

    const selectedCount = profiles.filter(p => p.selected).length;
    compareBtn.disabled = selectedCount < 2;
}

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value);
    return div.innerHTML;
}

function toggleProfileSelected(id, selected) {
    const profiles = loadSavedProfiles();
    const updated = profiles.map(p => (p.id === id ? { ...p, selected } : p));
    persistSavedProfiles(updated);
    renderSavedProfiles();
    if (compareTable.innerHTML) {
        buildCompareTable();
    }
}

function removeSavedProfile(id) {
    const profiles = loadSavedProfiles().filter(p => p.id !== id);
    persistSavedProfiles(profiles);
    renderSavedProfiles();
    buildCompareTable();
}

saveProfileBtn.addEventListener("click", function () {
    if (!lastEmployeeData || !lastPredictionResult) {
        saveSuccess.textContent = "";
        formError.textContent = "Please predict your salary first.";
        return;
    }

    const profiles = loadSavedProfiles();
    const newProfile = {
        id: `p_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        name: `${lastEmployeeData.occupation} · ${lastEmployeeData.country}`,
        savedAt: new Date().toISOString(),
        employee: lastEmployeeData,
        prediction: lastPredictionResult,
        selected: false,
    };

    profiles.unshift(newProfile);
    // Cap the list so local storage doesn't grow without bound
    const trimmed = profiles.slice(0, MAX_SAVED_PROFILES);
    persistSavedProfiles(trimmed);
    renderSavedProfiles();

    saveProfileBtn.classList.add("saved");
    saveProfileLabel.textContent = "Saved ✓";
    saveSuccess.textContent = "Profile saved below — scroll down to compare it with others.";
});

compareBtn.addEventListener("click", function () {
    buildCompareTable();
});

function buildCompareTable() {
    const selected = loadSavedProfiles().filter(p => p.selected);

    if (selected.length < 2) {
        compareTableWrap.classList.add("hidden");
        compareTable.innerHTML = "";
        return;
    }

    const rows = [
        { label: "Predicted salary", get: p => formatCurrency(p.prediction.predicted_salary), cls: "salary-cell" },
        { label: "Estimated range", get: p => `${formatCurrency(p.prediction.salary_range.minimum)} – ${formatCurrency(p.prediction.salary_range.maximum)}` },
        { label: "Occupation", get: p => p.employee.occupation },
        { label: "Country", get: p => p.employee.country },
        { label: "Industry", get: p => p.employee.industry },
        { label: "Education", get: p => p.employee.education_level },
        { label: "Experience", get: p => `${p.employee.years_of_experience} yrs` },
        { label: "Hours / week", get: p => p.employee.hours_per_week },
        { label: "Experience level", get: p => p.prediction.experience_level },
        { label: "Salary position", get: p => p.prediction.salary_insight },
    ];

    const maxSalary = Math.max(...selected.map(p => p.prediction.predicted_salary));

    let html = "<thead><tr><th>Metric</th>";
    selected.forEach(p => (html += `<th>${escapeHtml(p.name)}</th>`));
    html += "</tr></thead><tbody>";

    rows.forEach(row => {
        html += `<tr><td class="metric-label">${row.label}</td>`;
        selected.forEach(p => {
            const isBestSalary = row.cls === "salary-cell" && p.prediction.predicted_salary === maxSalary && selected.length > 1;
            const cellClass = isBestSalary ? "best-cell" : (row.cls || "");
            html += `<td class="${cellClass}">${escapeHtml(row.get(p))}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody>";

    compareTable.innerHTML = html;
    compareTableWrap.classList.remove("hidden");
}

// Render any previously saved profiles as soon as the page loads
renderSavedProfiles();
