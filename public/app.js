// Chart instances container
let charts = {
  price: null,
  rsi: null,
  macd: null,
  feature: null,
  backtest: null
};

let currentPipelineData = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadSynopsis();
  loadStockData("AAPL", "2y");
});

function setupEventListeners() {
  const analyzeBtn = document.getElementById("analyze-btn");
  const heroDemoBtn = document.getElementById("hero-demo-btn");
  const tickerInput = document.getElementById("ticker-input");
  const periodSelect = document.getElementById("period-select");
  const quickPills = document.querySelectorAll(".pill-btn");

  analyzeBtn.addEventListener("click", () => {
    const symbol = tickerInput.value.trim() || "AAPL";
    loadStockData(symbol, periodSelect.value);
  });

  if (heroDemoBtn) {
    heroDemoBtn.addEventListener("click", () => {
      tickerInput.value = "AAPL";
      loadStockData("AAPL", periodSelect.value);
      document.getElementById("terminal-section").scrollIntoView({ behavior: 'smooth' });
    });
  }

  tickerInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      const symbol = tickerInput.value.trim() || "AAPL";
      loadStockData(symbol, periodSelect.value);
    }
  });

  periodSelect.addEventListener("change", () => {
    const symbol = tickerInput.value.trim() || "AAPL";
    loadStockData(symbol, periodSelect.value);
  });

  quickPills.forEach(pill => {
    pill.addEventListener("click", () => {
      quickPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const symbol = pill.getAttribute("data-symbol");
      tickerInput.value = symbol;
      loadStockData(symbol, periodSelect.value);
    });
  });

  // Chart toggles
  document.getElementById("toggle-sma20").addEventListener("change", updatePriceChartToggles);
  document.getElementById("toggle-sma50").addEventListener("change", updatePriceChartToggles);
  document.getElementById("toggle-bb").addEventListener("change", updatePriceChartToggles);
}

async function loadSynopsis() {
  const box = document.getElementById("synopsis-box");
  try {
    const res = await fetch("/api/synopsis");
    if (res.ok) {
      const data = await res.json();
      if (data.synopsis && window.marked) {
        box.innerHTML = marked.parse(data.synopsis);
      }
    }
  } catch (e) {
    console.warn("Synopsis fetch skipped:", e);
  }
}

function setLoadingState(isLoading) {
  const btnText = document.querySelector("#analyze-btn .btn-text");
  const spinner = document.getElementById("btn-spinner");
  const analyzeBtn = document.getElementById("analyze-btn");

  if (isLoading) {
    btnText.textContent = "Processing...";
    spinner.classList.remove("hidden");
    analyzeBtn.disabled = true;
  } else {
    btnText.textContent = "Predict";
    spinner.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
}

async function loadStockData(symbol, period) {
  setLoadingState(true);
  try {
    const response = await fetch(`/api/analyze?symbol=${encodeURIComponent(symbol)}&period=${period}`);
    if (!response.ok) {
      throw new Error(`HTTP Error ${response.status}`);
    }
    const data = await response.json();
    currentPipelineData = data;
    
    renderDashboard(data);
  } catch (error) {
    console.error("Failed to fetch ML stock analysis:", error);
    alert(`Failed to analyze stock ticker ${symbol}. Please verify the symbol or try again.`);
  } finally {
    setLoadingState(false);
  }
}

function renderDashboard(data) {
  // 1. Asset Banner Update
  document.getElementById("asset-symbol").textContent = data.symbol;
  document.getElementById("asset-name").textContent = getAssetName(data.symbol);
  document.getElementById("asset-price").textContent = `$${data.current_price.toFixed(2)}`;
  document.getElementById("asset-date").textContent = `As of ${data.latest_date}`;

  // Signal Badge
  const signalCard = document.getElementById("signal-value");
  signalCard.textContent = data.signal;
  signalCard.className = `signal-value badge-${data.signal.toLowerCase()}`;

  const ensMetrics = data.metrics.Ensemble || {};
  document.getElementById("metric-accuracy").textContent = `${ensMetrics.Directional_Accuracy || 50}%`;
  document.getElementById("metric-r2").textContent = ensMetrics.R2 !== undefined ? ensMetrics.R2 : "N/A";

  const day30Pred = data.future_predictions.find(p => p.day === 30) || data.future_predictions[data.future_predictions.length - 1];
  const forecastRetEl = document.getElementById("metric-forecast-return");
  if (day30Pred) {
    const retVal = day30Pred.expected_return_pct;
    forecastRetEl.textContent = `${retVal >= 0 ? '+' : ''}${retVal}%`;
    forecastRetEl.className = `mini-val ${retVal >= 0 ? 'positive' : 'negative'}`;
  }

  // 2. Render Charts
  renderPriceChart(data.historical_data);
  renderRSIChart(data.historical_data);
  renderMACDChart(data.historical_data);
  renderFeatureImportanceChart(data.feature_importances);
  renderBacktestChart(data.backtest.equity_curve);

  // 3. Render Tables
  renderForecastTable(data.future_predictions);
  renderModelsTable(data.metrics);

  // 4. Render Backtest Metrics
  renderBacktestMetrics(data.backtest);
}

function getAssetName(symbol) {
  const names = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "SPY": "SPDR S&P 500 ETF Trust",
    "BTC-USD": "Bitcoin USD"
  };
  return names[symbol.toUpperCase()] || `${symbol.toUpperCase()} Asset`;
}

/* --- CHART 1: Price & Technical Indicators --- */
function renderPriceChart(historicalData) {
  const ctx = document.getElementById("priceChart").getContext("2d");
  const labels = historicalData.map(d => d.date);
  const prices = historicalData.map(d => d.close);
  const sma20 = historicalData.map(d => d.sma_20);
  const sma50 = historicalData.map(d => d.sma_50);
  const bbUpper = historicalData.map(d => d.bb_upper);
  const bbLower = historicalData.map(d => d.bb_lower);

  if (charts.price) charts.price.destroy();

  charts.price = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Price ($)",
          data: prices,
          borderColor: "#00f2fe",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          yAxisID: "y"
        },
        {
          label: "SMA 20",
          data: sma20,
          borderColor: "#38bdf8",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          yAxisID: "y"
        },
        {
          label: "SMA 50",
          data: sma50,
          borderColor: "#fbbf24",
          borderWidth: 1.5,
          pointRadius: 0,
          yAxisID: "y"
        },
        {
          label: "BB Upper",
          data: bbUpper,
          borderColor: "rgba(139, 92, 246, 0.4)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          yAxisID: "y"
        },
        {
          label: "BB Lower",
          data: bbLower,
          borderColor: "rgba(139, 92, 246, 0.4)",
          borderWidth: 1,
          pointRadius: 0,
          fill: "-1",
          backgroundColor: "rgba(139, 92, 246, 0.06)",
          yAxisID: "y"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { family: "Inter" } } },
        tooltip: { backgroundColor: "rgba(15, 23, 42, 0.9)", titleColor: "#00f2fe" }
      },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function updatePriceChartToggles() {
  if (!charts.price) return;
  const showSma20 = document.getElementById("toggle-sma20").checked;
  const showSma50 = document.getElementById("toggle-sma50").checked;
  const showBB = document.getElementById("toggle-bb").checked;

  charts.price.data.datasets[1].hidden = !showSma20;
  charts.price.data.datasets[2].hidden = !showSma50;
  charts.price.data.datasets[3].hidden = !showBB;
  charts.price.data.datasets[4].hidden = !showBB;
  charts.price.update();
}

/* --- CHART 2: RSI Oscillator --- */
function renderRSIChart(historicalData) {
  const ctx = document.getElementById("rsiChart").getContext("2d");
  const labels = historicalData.map(d => d.date);
  const rsi = historicalData.map(d => d.rsi_14);

  if (charts.rsi) charts.rsi.destroy();

  charts.rsi = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: "RSI (14)",
        data: rsi,
        borderColor: "#8b5cf6",
        borderWidth: 1.5,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.03)" } },
        y: { min: 0, max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

/* --- CHART 3: MACD Oscillator --- */
function renderMACDChart(historicalData) {
  const ctx = document.getElementById("macdChart").getContext("2d");
  const labels = historicalData.map(d => d.date);
  const macdLine = historicalData.map(d => d.macd_line);
  const macdSignal = historicalData.map(d => d.macd_signal);

  if (charts.macd) charts.macd.destroy();

  charts.macd = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "MACD Line", data: macdLine, borderColor: "#38bdf8", borderWidth: 1.5, pointRadius: 0 },
        { label: "Signal Line", data: macdSignal, borderColor: "#f43f5e", borderWidth: 1.5, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8", font: { size: 10 } } } },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.03)" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

/* --- CHART 4: Feature Importance --- */
function renderFeatureImportanceChart(importances) {
  const ctx = document.getElementById("featureImportanceChart").getContext("2d");
  const keys = Object.keys(importances).slice(0, 8);
  const values = keys.map(k => importances[k]);

  if (charts.feature) charts.feature.destroy();

  charts.feature = new Chart(ctx, {
    type: "bar",
    data: {
      labels: keys,
      datasets: [{
        label: "Importance (%)",
        data: values,
        backgroundColor: "rgba(0, 242, 254, 0.65)",
        borderColor: "#00f2fe",
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#f8fafc", font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
}

/* --- CHART 5: Backtest Equity Curve --- */
function renderBacktestChart(equityCurve) {
  const ctx = document.getElementById("backtestChart").getContext("2d");
  const labels = equityCurve.map(d => d.date);
  const stratValues = equityCurve.map(d => d.strategy);
  const bhValues = equityCurve.map(d => d.buy_and_hold);

  if (charts.backtest) charts.backtest.destroy();

  charts.backtest = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "ML Strategy Portfolio ($)", data: stratValues, borderColor: "#10b981", borderWidth: 2, pointRadius: 0 },
        { label: "Buy & Hold Benchmark ($)", data: bhValues, borderColor: "#64748b", borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        x: { ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

/* --- TABLES & METRICS RENDERING --- */
function renderForecastTable(predictions) {
  const tbody = document.getElementById("forecast-table-body");
  tbody.innerHTML = "";

  const horizons = [1, 3, 5, 10, 15, 20, 30];
  const filtered = predictions.filter(p => horizons.includes(p.day));

  filtered.forEach(item => {
    const tr = document.createElement("tr");
    const retClass = item.expected_return_pct >= 0 ? "positive" : "negative";
    const sign = item.expected_return_pct >= 0 ? "+" : "";

    tr.innerHTML = `
      <td><strong>Day ${item.day}</strong></td>
      <td><strong>$${item.predicted_price.toFixed(2)}</strong></td>
      <td class="${retClass}"><strong>${sign}${item.expected_return_pct}%</strong></td>
      <td style="color:#64748b;">$${item.lower_bound.toFixed(2)}</td>
      <td style="color:#64748b;">$${item.upper_bound.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderModelsTable(metrics) {
  const tbody = document.getElementById("models-table-body");
  tbody.innerHTML = "";

  Object.keys(metrics).forEach(modelName => {
    const m = metrics[modelName];
    const tr = document.createElement("tr");
    const isEnsemble = modelName === "Ensemble";

    tr.innerHTML = `
      <td><strong>${isEnsemble ? '⭐ Ensemble' : modelName}</strong></td>
      <td class="positive"><strong>${m.Directional_Accuracy}%</strong></td>
      <td>${m.RMSE}</td>
      <td>${m.MAE}</td>
      <td>${m.R2}</td>
    `;
    if (isEnsemble) {
      tr.style.background = "rgba(0, 242, 254, 0.08)";
    }
    tbody.appendChild(tr);
  });
}

function renderBacktestMetrics(backtest) {
  document.getElementById("bt-roi").textContent = `${backtest.strategy_roi_pct >= 0 ? '+' : ''}${backtest.strategy_roi_pct}%`;
  document.getElementById("bt-roi").className = `m-value ${backtest.strategy_roi_pct >= 0 ? 'positive' : 'negative'}`;

  document.getElementById("bt-bh-roi").textContent = `${backtest.buy_hold_roi_pct >= 0 ? '+' : ''}${backtest.buy_hold_roi_pct}%`;
  document.getElementById("bt-sharpe").textContent = backtest.sharpe_ratio;

  document.getElementById("bt-mdd").textContent = `${backtest.max_drawdown_pct}%`;
  document.getElementById("bt-winrate").textContent = `${backtest.win_rate_pct}%`;
}
