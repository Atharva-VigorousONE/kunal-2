// Globals
let metricsData = null;
let activeProduct = 'overall';

// Run on page load
document.addEventListener('DOMContentLoaded', () => {
    fetchMetrics();
    setupEventListeners();
});

// Setup event listeners for tabs and templates
function setupEventListeners() {
    // Product Navigation Selector
    const navButtons = document.querySelectorAll('#productSelectorNav .selector-pill');
    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeProduct = btn.dataset.product;
            updateDashboardView();
        });
    });

    // Sandbox Templates Click
    const templateBtns = document.querySelectorAll('#sandboxTemplates .template-btn');
    templateBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.dataset.text;
            document.getElementById('reviewInputText').value = text;
        });
    });
}

// Fetch Metrics from Backend API
async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        metricsData = await response.ok ? await response.json() : null;
        if (metricsData) {
            updateDashboardView();
            renderWordImportance();
            populateExplorerTable();
        }
    } catch (error) {
        console.error("Failed to load metrics data:", error);
    }
}

// Tab Switching System
function switchTab(tabName) {
    // Toggle active tab buttons
    document.getElementById('tabBtnPlayground').classList.toggle('active', tabName === 'playground');
    document.getElementById('tabBtnUnreliability').classList.toggle('active', tabName === 'unreliability');
    document.getElementById('tabBtnExplorer').classList.toggle('active', tabName === 'explorer');

    // Toggle active tab content panels
    document.getElementById('tabContentPlayground').classList.toggle('active', tabName === 'playground');
    document.getElementById('tabContentUnreliability').classList.toggle('active', tabName === 'unreliability');
    document.getElementById('tabContentExplorer').classList.toggle('active', tabName === 'explorer');
    
    if (tabName === 'explorer') {
        populateExplorerTable();
    }
}

// Update KPI Metrics and Confusion Matrix based on Active Product Selection
function updateDashboardView() {
    if (!metricsData) return;

    let targetObj = null;
    let labelText = '';

    if (activeProduct === 'overall') {
        targetObj = metricsData.overall;
        labelText = `${targetObj.total_samples} total test samples`;
    } else {
        targetObj = metricsData.products[activeProduct];
        labelText = `${targetObj.total_samples} total product reviews`;
    }

    if (!targetObj) return;

    const metrics = targetObj.metrics;

    // Update KPIs
    document.getElementById('val-accuracy').innerText = (metrics.accuracy * 100).toFixed(2) + '%';
    document.getElementById('val-precision').innerText = (metrics.precision * 100).toFixed(2) + '%';
    document.getElementById('val-recall').innerText = (metrics.recall * 100).toFixed(2) + '%';
    document.getElementById('val-f1').innerText = (metrics.f1_score * 100).toFixed(2) + '%';
    document.getElementById('lbl-samples-count').innerText = labelText;

    // Update Confusion Matrix Box Numbers
    const cm = metrics.confusion_matrix;
    document.getElementById('val-tn').innerText = cm[0][0];
    document.getElementById('val-fp').innerText = cm[0][1];
    document.getElementById('val-fn').innerText = cm[1][0];
    document.getElementById('val-tp').innerText = cm[1][1];
}

// Render Top Word Features Coefficient Bar Charts
function renderWordImportance() {
    if (!metricsData) return;

    const container = document.getElementById('barChartContainer');
    container.innerHTML = '';

    // Take top 5 positive and top 5 negative words
    const negWords = metricsData.top_negative_words.slice(0, 5);
    const posWords = metricsData.top_positive_words.slice(0, 5);

    // Merge them: negatives sorted descending (most negative first), then positives
    const displayWords = [...negWords, ...posWords];

    // Find maximum absolute coefficient value for percentage scaling
    const maxVal = Math.max(...displayWords.map(w => Math.abs(w.coefficient)));

    displayWords.forEach(item => {
        const word = item.word;
        const coef = item.coefficient;
        const isPos = coef >= 0;
        
        // Calculate percentage width (with min width of 5% for visibility)
        const pctWidth = Math.max(5, (Math.abs(coef) / maxVal) * 100);

        // Build HTML row
        const row = document.createElement('div');
        row.className = 'bar-row';

        const label = document.createElement('div');
        label.className = 'bar-label';
        label.innerText = word;
        label.title = word;

        const track = document.createElement('div');
        track.className = 'bar-track';

        const fill = document.createElement('div');
        fill.className = `bar-fill ${isPos ? 'pos' : 'neg'}`;
        
        // We float right for negative coefficients, left for positive
        fill.style.width = pctWidth + '%';
        if (isPos) {
            fill.style.left = '0';
            fill.style.position = 'absolute';
        } else {
            fill.style.right = '0';
            fill.style.position = 'absolute';
        }

        const value = document.createElement('div');
        value.className = 'bar-value';
        value.innerText = coef.toFixed(2);

        track.appendChild(fill);
        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        container.appendChild(row);
    });
}

// Sandbox: Analyze Review Sentiment in Real-time
async function analyzeSentiment() {
    const text = document.getElementById('reviewInputText').value.trim();
    if (!text) return;

    const btn = document.getElementById('btnAnalyze');
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        if (!response.ok) {
            throw new Error(`Inference request failed with status: ${response.status}`);
        }

        const result = await response.json();
        renderSandboxResult(result);

    } catch (error) {
        console.error("Error during real-time analysis:", error);
        alert("Server inference failed. Please ensure backend is running.");
    } finally {
        btn.innerText = 'Analyze Review';
        btn.disabled = false;
    }
}

// Render Sandbox Output Visuals
function renderSandboxResult(result) {
    const container = document.getElementById('sandboxResultContainer');
    container.style.display = 'block';

    const isPos = result.prediction === 'positive';
    
    // Set Sentiment Badge
    const badge = document.getElementById('badgeSentiment');
    badge.innerText = isPos ? 'Positive Sentiment' : 'Negative Sentiment';
    badge.className = `badge ${isPos ? 'badge-positive' : 'badge-negative'}`;

    // Set Confidence Score
    const confidencePct = (result.probability * 100).toFixed(1);
    document.getElementById('lblConfidence').innerText = confidencePct + '%';

    // Set Confidence progress bar width and class
    const fill = document.getElementById('barConfidenceFill');
    fill.style.width = confidencePct + '%';
    fill.className = `confidence-fill ${isPos ? 'positive' : 'negative'}`;

    // Highlight Words based on contribution
    const textRenderArea = document.getElementById('explainTextRendered');
    textRenderArea.innerHTML = '';

    result.word_contributions.forEach(item => {
        const token = item.word;
        const contrib = item.contribution;
        const coef = item.weight;
        const tfidf = item.tfidf;

        const span = document.createElement('span');
        span.className = 'explain-word';
        span.innerText = token + ' ';

        // Color coding logic
        if (contrib > 0.005) {
            // Positive contribution (green background opacity scaled)
            // Cap opacity at 0.8 to ensure text is readable
            const opacity = Math.min(0.8, contrib * 5.0); 
            span.style.backgroundColor = `rgba(16, 185, 129, ${opacity})`;
            span.style.color = '#ffffff';
        } else if (contrib < -0.005) {
            // Negative contribution (red background opacity scaled)
            const opacity = Math.min(0.8, Math.abs(contrib) * 5.0);
            span.style.backgroundColor = `rgba(244, 63, 94, ${opacity})`;
            span.style.color = '#ffffff';
        } else {
            // Negligible contribution (no background, muted text)
            span.style.color = 'var(--text-primary)';
        }

        // Feature Tooltip on hover
        const tooltip = document.createElement('span');
        tooltip.className = 'tooltip-weight';
        tooltip.innerHTML = `Word: <strong>${token.toLowerCase()}</strong><br>
                             Weight (coef): ${coef.toFixed(3)}<br>
                             TF-IDF: ${tfidf.toFixed(3)}<br>
                             Contrib: <strong>${contrib.toFixed(4)}</strong>`;
        span.appendChild(tooltip);
        textRenderArea.appendChild(span);
    });

    // Render Warnings Engine alert box
    const warningBox = document.getElementById('warningBox');
    warningBox.innerHTML = '';
    
    if (result.unreliability_warnings.length > 0) {
        warningBox.style.display = 'flex';
        result.unreliability_warnings.forEach(warn => {
            const warnItem = document.createElement('div');
            warnItem.className = 'warning-item';
            warnItem.innerText = warn;
            warningBox.appendChild(warnItem);
        });
    } else {
        warningBox.style.display = 'none';
    }
}

// Dataset Explorer Table population
function populateExplorerTable() {
    if (!metricsData) return;

    const errorType = document.getElementById('explorerFilterErrorType').value;
    const filterProd = document.getElementById('explorerFilterProduct').value;
    
    const cases = metricsData[errorType];
    const tbody = document.getElementById('explorerTableBody');
    tbody.innerHTML = '';

    if (!cases || cases.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-muted);">No records found matching this analysis type.</td>`;
        tbody.appendChild(row);
        return;
    }

    let renderedCount = 0;

    cases.forEach(item => {
        // Apply product filter
        if (filterProd !== 'all' && item.product !== filterProd) {
            return;
        }

        renderedCount++;

        const tr = document.createElement('tr');

        // Product cell
        const tdProd = document.createElement('td');
        tdProd.className = 'explorer-product-col';
        tdProd.innerText = item.product.split(' - ')[0]; // truncate long product names
        tdProd.title = item.product;

        // Review Text cell
        const tdText = document.createElement('td');
        tdText.className = 'explorer-text-col';
        tdText.innerText = item.message;

        // Rating cell
        const tdRating = document.createElement('td');
        tdRating.className = 'explorer-rating-col';
        tdRating.style.textAlign = 'center';
        tdRating.innerText = item.rating.toFixed(1);
        
        // Style rating coloring
        if (item.rating >= 4.0) tdRating.style.color = 'var(--color-positive)';
        else if (item.rating <= 2.0) tdRating.style.color = 'var(--color-negative)';
        else tdRating.style.color = 'var(--color-neutral)';

        // Confidence / Predicted Cell
        const tdConfidence = document.createElement('td');
        tdConfidence.style.textAlign = 'center';
        
        let labelText = '';
        if (errorType === 'false_positives') {
            labelText = `<span style="color: var(--color-positive); font-weight: 500;">FP</span><br>
                         <span style="font-size: 0.7rem; color: var(--text-muted);">Prob: ${(item.probability * 100).toFixed(0)}%</span>`;
        } else if (errorType === 'false_negatives') {
            labelText = `<span style="color: var(--color-negative); font-weight: 500;">FN</span><br>
                         <span style="font-size: 0.7rem; color: var(--text-muted);">Prob: ${(item.probability * 100).toFixed(0)}%</span>`;
        } else {
            labelText = `<span style="color: ${item.predicted_label === 'positive' ? 'var(--color-positive)' : 'var(--color-negative)'}; font-weight: 500;">
                         ${item.predicted_label.toUpperCase()}</span><br>
                         <span style="font-size: 0.7rem; color: var(--text-muted);">Prob: ${(item.probability * 100).toFixed(0)}%</span>`;
        }
        tdConfidence.innerHTML = labelText;

        // Analyst Explanation cell
        const tdExplanation = document.createElement('td');
        tdExplanation.style.fontSize = '0.8rem';
        tdExplanation.style.color = 'var(--text-secondary)';
        tdExplanation.innerText = item.analysis;

        tr.appendChild(tdProd);
        tr.appendChild(tdText);
        tr.appendChild(tdRating);
        tr.appendChild(tdConfidence);
        tr.appendChild(tdExplanation);
        tbody.appendChild(tr);
    });

    if (renderedCount === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-muted);">No records found matching product filter.</td>`;
        tbody.appendChild(row);
    }
}

// Triggered when search filters are changed
function filterExplorerTable() {
    populateExplorerTable();
}

// Helper to load review templates directly into playground
function loadAndAnalyze(text) {
    document.getElementById('reviewInputText').value = text;
    switchTab('playground');
    analyzeSentiment();
}
