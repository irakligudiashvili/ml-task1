from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import cost_benefit_analysis as cba
import price_prediction as pp

app = FastAPI(title="AutoValuation ML API")


@app.on_event("startup")
def load_and_train_model():
    pp.train_model()


@app.post("/api/v1/predict")
def predict_price(car: pp.CarFeatures):
    prediction_result = pp.predict(car)
    prediction_result["feature_coefficients"] = cba.get_feature_coefficients()
    return prediction_result


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --primary: #4f46e5;
                --primary-light: #6366f1;
                --text-dark: #0f172a;
                --text-muted: #64748b;
                --border: #e2e8f0;
                --input-bg: #f8fafc;
                --dark-card: #0f172a;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { background-color: var(--bg); color: var(--text-dark); padding: 24px; min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 32px; }
            .layout-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 24px; }
            .card { background: var(--card-bg); border-radius: 16px; padding: 28px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
            .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .section-title { font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
            .reset-link { color: var(--primary); font-size: 13px; font-weight: 500; text-decoration: none; cursor: pointer; }
            .sub-title { font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; margin: 20px 0 12px; }
            .preset-group { display: flex; gap: 12px; flex-wrap: wrap; }
            .preset-btn { background: var(--input-bg); border: 1px solid var(--border); padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 6px; }
            .preset-btn:hover { border-color: var(--primary); }
            .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
            .form-group { display: flex; flex-direction: column; gap: 6px; }
            .form-group label { font-size: 12px; font-weight: 500; color: var(--text-muted); }
            select, input[type="text"], input[type="number"] { background: var(--input-bg); border: 1px solid var(--border); padding: 10px 12px; border-radius: 8px; font-size: 13px; width: 100%; }
            .slider-group { margin-bottom: 16px; }
            .slider-header { display: flex; justify-content: space-between; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
            .slider-header span.val { color: var(--primary); font-weight: 700; }
            input[type="range"] { -webkit-appearance: none; width: 100%; height: 6px; background: #e2e8f0; border-radius: 3px; outline: none; }
            input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--primary); cursor: pointer; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
            .valuation-card { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: white; border-radius: 16px; padding: 24px; position: relative; margin-bottom: 20px; }
            .val-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.7; font-weight: 600; }
            .val-amount { font-size: 42px; font-weight: 800; margin: 12px 0 4px; }
            .val-range { font-size: 12px; opacity: 0.8; margin-bottom: 20px; }
            .progress-bar-bg { background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 8px; }
            .progress-bar-fill { background: #6366f1; height: 100%; width: 68%; }
            .drivers-card { margin-bottom: 20px; }
            .driver-item { display: flex; justify-content: space-between; font-size: 13px; padding: 10px 0; border-bottom: 1px solid var(--border); }
            .driver-item:last-child { border-bottom: none; }
            .driver-val { font-weight: 700; color: #ef4444; }
            .driver-val.positive { color: #10b981; }
            .compromise-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
            .impact-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }
            .impact-table th, .impact-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
            .impact-table th { background: #f1f5f9; color: var(--text-muted); font-weight: 600; }
            .compromise-badge { padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
            .badge-high { background: #dcfce7; color: #15803d; }
            .badge-med { background: #fef9c3; color: #a16207; }
            .badge-low { background: #fee2e2; color: #b91c1c; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="layout-grid">
                <div class="card">
                    <div class="section-header">
                        <div class="section-title">⚙️ Vehicle Price Predictor</div>
                        <a class="reset-link" onclick="resetDefaults()">↺ Reset Defaults</a>
                    </div>
                    <div class="sub-title">Quick Presets</div>
                    <div class="preset-group">
                        <button class="preset-btn" onclick="applyPreset('toyota')">🚗 Toyota Sedan</button>
                        <button class="preset-btn" onclick="applyPreset('porsche')">🏎️ Porsche Sports</button>
                        <button class="preset-btn" onclick="applyPreset('vw')">🚙 VW Diesel Hatch</button>
                    </div>
                    <div class="sub-title">Categorical Metadata</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Brand</label>
                            <select id="CarName" onchange="updateAll()">
                                <option value="toyota">Toyota</option>
                                <option value="porsche">Porsche</option>
                                <option value="volkswagen">Volkswagen</option>
                                <option value="bmw">BMW</option>
                                <option value="honda">Honda</option>
                                <option value="mazda">Mazda</option>
                                <option value="nissan">Nissan</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Body Style</label>
                            <select id="carbody" onchange="updateAll()">
                                <option value="sedan">Sedan</option>
                                <option value="hatchback">Hatchback</option>
                                <option value="wagon">Wagon</option>
                                <option value="hardtop">Hardtop</option>
                                <option value="convertible">Convertible</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Fuel Type</label>
                            <select id="fueltype" onchange="updateAll()">
                                <option value="gas">Gasoline</option>
                                <option value="diesel">Diesel</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Aspiration</label>
                            <select id="aspiration" onchange="updateAll()">
                                <option value="std">Standard</option>
                                <option value="turbo">Turbo</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Drive Wheel</label>
                            <select id="drivewheel" onchange="updateAll()">
                                <option value="fwd">Front-Wheel Drive</option>
                                <option value="rwd">Rear-Wheel Drive</option>
                                <option value="4wd">Four-Wheel Drive</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Engine Location</label>
                            <select id="enginelocation" onchange="updateAll()">
                                <option value="front">Front</option>
                                <option value="rear">Rear</option>
                            </select>
                        </div>
                    </div>
                    <div class="sub-title">Engine & Performance Metrics</div>
                    <div class="slider-group">
                        <div class="slider-header">
                            <span>Engine Size (cu. in.)</span>
                            <span class="val" id="val-enginesize">109</span>
                        </div>
                        <input type="range" id="enginesize" min="60" max="326" value="109" oninput="updateAll()">
                    </div>
                    <div class="slider-group">
                        <div class="slider-header">
                            <span>Horsepower (HP)</span>
                            <span class="val" id="val-horsepower">102</span>
                        </div>
                        <input type="range" id="horsepower" min="48" max="288" value="102" oninput="updateAll()">
                    </div>
                    <div class="slider-group">
                        <div class="slider-header">
                            <span>Curb Weight (lbs)</span>
                            <span class="val" id="val-curbweight">2335</span>
                        </div>
                        <input type="range" id="curbweight" min="1488" max="4066" value="2335" oninput="updateAll()">
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>City MPG</span>
                                <span class="val" id="val-citympg">24</span>
                            </div>
                            <input type="range" id="citympg" min="13" max="49" value="24" oninput="updateAll()">
                        </div>
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Highway MPG</span>
                                <span class="val" id="val-highwaympg">30</span>
                            </div>
                            <input type="range" id="highwaympg" min="16" max="54" value="30" oninput="updateAll()">
                        </div>
                    </div>
                    <div class="form-grid" style="margin-top: 10px;">
                        <div class="form-group">
                            <label>Wheelbase</label>
                            <input type="number" id="wheelbase" value="98.4" step="0.1" onchange="updateAll()">
                        </div>
                        <div class="form-group">
                            <label>Car Length</label>
                            <input type="number" id="carlength" value="176.9" step="0.1" onchange="updateAll()">
                        </div>
                        <div class="form-group">
                            <label>Car Width</label>
                            <input type="number" id="carwidth" value="65.4" step="0.1" onchange="updateAll()">
                        </div>
                    </div>
                </div>
                <div>
                    <div class="valuation-card">
                        <div class="val-title">Estimated Market Value</div>
                        <div class="val-amount" id="res-price">$10,152</div>
                        <div class="val-range">95% Confidence Interval: <span id="res-range">$8,934 – $11,370</span></div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px; opacity: 0.7;">
                            <span>Market Baseline Avg:</span>
                            <span>$13,276</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" id="res-bar"></div>
                        </div>
                    </div>
                    <div class="card drivers-card">
                        <div class="section-title" style="margin-bottom: 6px; font-size: 15px;">📊 Key Price Drivers</div>
                        <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Estimated value impact breakdown for selected features:</p>
                        <div class="driver-item">
                            <span id="driver-brand-label">Brand Value (TOYOTA)</span>
                            <span class="driver-val positive" id="driver-brand">+$400</span>
                        </div>
                        <div class="driver-item">
                            <span id="driver-engine-label">Engine Size (109 cu in)</span>
                            <span class="driver-val positive" id="driver-engine">+$1,622</span>
                        </div>
                        <div class="driver-item">
                            <span id="driver-hp-label">Horsepower Output (102 HP)</span>
                            <span class="driver-val positive" id="driver-hp">+$136</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="section-title" style="margin-bottom: 8px;">📈 Cost vs. Value & Compromise Analyzer</div>
                <div class="compromise-grid">
                    <div>
                        <table class="impact-table">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Dollar Impact / Unit</th>
                                    <th>Compromise Viability</th>
                                </tr>
                            </thead>
                            <tbody id="coef-table-body"></tbody>
                        </table>
                    </div>
                    <div>
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Engine Size Reduction</span>
                                <span class="val" id="val-cut-engine">20 cu in</span>
                            </div>
                            <input type="range" id="cut-engine" min="0" max="80" value="20" oninput="calculateSavings()">
                        </div>
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Horsepower Reduction</span>
                                <span class="val" id="val-cut-hp">15 HP</span>
                            </div>
                            <input type="range" id="cut-hp" min="0" max="60" value="15" oninput="calculateSavings()">
                        </div>
                        <div class="slider-group">
                            <div class="slider-header">
                                <span>Weight Reduction</span>
                                <span class="val" id="val-cut-weight">200 lbs</span>
                            </div>
                            <input type="range" id="cut-weight" min="0" max="500" step="50" value="200" oninput="calculateSavings()">
                        </div>
                        <div style="background: #f1f5f9; padding: 20px; border-radius: 12px; margin-top: 16px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 11px; color: var(--text-muted); font-weight: 700; text-transform: uppercase;">Estimated Price Reduction</div>
                                <div style="font-size: 30px; font-weight: 800; color: #10b981; margin-top: 4px;" id="total-savings">-$0</div>
                            </div>
                            <div>
                                <div style="font-size: 11px; color: var(--text-muted); font-weight: 700; text-transform: uppercase;">Adjusted Valuation</div>
                                <div style="font-size: 20px; font-weight: 700; color: var(--primary); margin-top: 4px;" id="adjusted-price">$0</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            let cachedCoefficients = {};
            let currentBasePrice = 0;

            function getPayload() {
                return {
                    wheelbase: parseFloat(document.getElementById('wheelbase').value),
                    carlength: parseFloat(document.getElementById('carlength').value),
                    carwidth: parseFloat(document.getElementById('carwidth').value),
                    carheight: 54.3,
                    curbweight: parseFloat(document.getElementById('curbweight').value),
                    enginesize: parseFloat(document.getElementById('enginesize').value),
                    boreratio: 3.19,
                    stroke: 3.40,
                    compressionratio: 10.0,
                    horsepower: parseFloat(document.getElementById('horsepower').value),
                    peakrpm: 5500,
                    citympg: parseFloat(document.getElementById('citympg').value),
                    highwaympg: parseFloat(document.getElementById('highwaympg').value),
                    CarName: document.getElementById('CarName').value,
                    fueltype: document.getElementById('fueltype').value,
                    aspiration: document.getElementById('aspiration').value,
                    doornumber: 'four',
                    carbody: document.getElementById('carbody').value,
                    drivewheel: document.getElementById('drivewheel').value,
                    enginelocation: document.getElementById('enginelocation').value,
                    enginetype: 'ohc',
                    cylindernumber: 'four',
                    fuelsystem: 'mpfi'
                };
            }

            function updateAll() {
                ['enginesize', 'horsepower', 'curbweight', 'citympg', 'highwaympg'].forEach(id => {
                    const el = document.getElementById(id);
                    if(el) document.getElementById('val-' + id).innerText = el.value;
                });
                fetchPrediction();
            }

            async function fetchPrediction() {
                const payload = getPayload();
                try {
                    const res = await fetch('/api/v1/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();

                    currentBasePrice = data.predicted_price;
                    document.getElementById('res-price').innerText = '$' + data.predicted_price.toLocaleString();
                    document.getElementById('res-range').innerText = '$' + data.confidence_lower.toLocaleString() + ' –$' + data.confidence_upper.toLocaleString();

                    const pct = Math.min(Math.max((data.predicted_price / 30000) * 100, 10), 100);
                    document.getElementById('res-bar').style.width = pct + '%';

                    document.getElementById('driver-brand-label').innerText = 'Brand Value (' + payload.CarName.toUpperCase() + ')';
                    document.getElementById('driver-brand').innerText = (data.brand_impact >= 0 ? '+$' : '-$') + Math.abs(data.brand_impact).toLocaleString();

                    document.getElementById('driver-engine-label').innerText = 'Engine Size (' + payload.enginesize + ' cu in)';
                    document.getElementById('driver-engine').innerText = (data.engine_impact >= 0 ? '+$' : '-$') + Math.abs(data.engine_impact).toLocaleString();

                    document.getElementById('driver-hp-label').innerText = 'Horsepower Output (' + payload.horsepower + ' HP)';
                    document.getElementById('driver-hp').innerText = (data.hp_impact >= 0 ? '+$' : '-$') + Math.abs(data.hp_impact).toLocaleString();

                    if(data.feature_coefficients) {
                        cachedCoefficients = data.feature_coefficients;
                        populateCoefTable(data.feature_coefficients);
                        calculateSavings();
                    }
                } catch(e) {
                    console.log(e);
                }
            }

            function populateCoefTable(coefs) {
                const tbody = document.getElementById('coef-table-body');
                if(!tbody || tbody.children.length > 0) return;

                const meta = [
                    { key: 'enginesize', label: 'Engine Size (cu in)', viability: 'High', class: 'badge-high' },
                    { key: 'horsepower', label: 'Horsepower (HP)', viability: 'High', class: 'badge-high' },
                    { key: 'curbweight', label: 'Curb Weight (lbs)', viability: 'Medium', class: 'badge-med' },
                    { key: 'carwidth', label: 'Car Width (in)', viability: 'Low', class: 'badge-low' },
                    { key: 'citympg', label: 'City MPG', viability: 'Low', class: 'badge-low' }
                ];

                meta.forEach(item => {
                    if(coefs[item.key] !== undefined) {
                        const tr = document.createElement('tr');
                        const val = coefs[item.key];
                        tr.innerHTML = `
                            <td><b>${item.label}</b></td>
                            <td style="color: ${val >= 0 ? '#10b981' : '#ef4444'}; font-weight: 700;">${val >= 0 ? '+$' : '-$'}${Math.abs(val)}</td>
                            <td><span class="compromise-badge ${item.class}">${item.viability}</span></td>
                        `;
                        tbody.appendChild(tr);
                    }
                });
            }

            function calculateSavings() {
                const cutEngine = parseFloat(document.getElementById('cut-engine').value);
                const cutHp = parseFloat(document.getElementById('cut-hp').value);
                const cutWeight = parseFloat(document.getElementById('cut-weight').value);

                document.getElementById('val-cut-engine').innerText = cutEngine + ' cu in';
                document.getElementById('val-cut-hp').innerText = cutHp + ' HP';
                document.getElementById('val-cut-weight').innerText = cutWeight + ' lbs';

                const engineCoef = cachedCoefficients['enginesize'] || 80.0;
                const hpCoef = cachedCoefficients['horsepower'] || 45.0;
                const weightCoef = cachedCoefficients['curbweight'] || 4.5;

                const totalSavings = (cutEngine * engineCoef) + (cutHp * hpCoef) + (cutWeight * weightCoef);
                const adjustedPrice = Math.max(0, currentBasePrice - totalSavings);

                document.getElementById('total-savings').innerText = '-$' + Math.round(totalSavings).toLocaleString();
                document.getElementById('adjusted-price').innerText = '$' + Math.round(adjustedPrice).toLocaleString();
            }

            function applyPreset(type) {
                if(type === 'toyota') {
                    document.getElementById('CarName').value = 'toyota';
                    document.getElementById('carbody').value = 'sedan';
                    document.getElementById('fueltype').value = 'gas';
                    document.getElementById('enginesize').value = 109;
                    document.getElementById('horsepower').value = 102;
                    document.getElementById('curbweight').value = 2335;
                } else if(type === 'porsche') {
                    document.getElementById('CarName').value = 'porsche';
                    document.getElementById('carbody').value = 'convertible';
                    document.getElementById('fueltype').value = 'gas';
                    document.getElementById('enginesize').value = 194;
                    document.getElementById('horsepower').value = 207;
                    document.getElementById('curbweight').value = 2756;
                } else if(type === 'vw') {
                    document.getElementById('CarName').value = 'volkswagen';
                    document.getElementById('carbody').value = 'hatchback';
                    document.getElementById('fueltype').value = 'diesel';
                    document.getElementById('enginesize').value = 97;
                    document.getElementById('horsepower').value = 68;
                    document.getElementById('curbweight').value = 2261;
                }
                updateAll();
            }

            function resetDefaults() {
                applyPreset('toyota');
            }

            window.onload = updateAll;
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)