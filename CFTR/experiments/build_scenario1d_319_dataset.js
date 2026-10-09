const fs = require('fs');

const input = 'YARISMA_TRAIN_CFTR_REDUCED.csv';
const output = 'YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv';
const metadata = 'YARISMA_TRAIN_CFTR_SCENARIO1D_319.preprocessing.json';

function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = ''; }
    else cell += ch;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

function escapeCsv(value) {
  const text = String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function quantile(values, probability) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * fraction;
}

const table = parseCsv(fs.readFileSync(input, 'utf8'));
const header = table[0];
const rows = table.slice(1).filter(row => row.length === header.length);
const index = Object.fromEntries(header.map((name, i) => [name, i]));
const groups = {
  GRUP_AL6_AL251: ['AL_6', 'AL_251'],
  GRUP_AL1_AL211: ['AL_1', 'AL_211'],
};
const removed = ['CAT_6', 'AL_6', 'AL_251', 'AL_1', 'AL_211'];
const parameters = {};

for (const [newName, members] of Object.entries(groups)) {
  parameters[newName] = {};
  for (const member of members) {
    const values = rows.map(row => Number(row[index[member]])).filter(Number.isFinite);
    const center = median(values);
    let scale = quantile(values, 0.75) - quantile(values, 0.25);
    if (!Number.isFinite(scale) || scale === 0) {
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      scale = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length) || 1;
    }
    parameters[newName][member] = { median: center, scale };
  }
}

const retained = header.filter(name => !removed.includes(name) && name !== 'Variant_ID' && name !== 'Label');
const featureOrder = [...retained, ...Object.keys(groups)];
const outputHeader = ['Variant_ID', 'Label', ...featureOrder];
const outputRows = [outputHeader];

for (const row of rows) {
  const generated = {};
  for (const [newName, members] of Object.entries(groups)) {
    const standardized = [];
    for (const member of members) {
      const value = Number(row[index[member]]);
      if (Number.isFinite(value)) {
        const parameter = parameters[newName][member];
        standardized.push((value - parameter.median) / parameter.scale);
      }
    }
    generated[newName] = standardized.length
      ? standardized.reduce((sum, value) => sum + value, 0) / standardized.length
      : '';
  }
  outputRows.push([
    row[index.Variant_ID], row[index.Label],
    ...retained.map(name => row[index[name]]),
    ...Object.keys(groups).map(name => generated[name]),
  ]);
}

fs.writeFileSync(output, outputRows.map(row => row.map(escapeCsv).join(',')).join('\n'));
fs.writeFileSync(metadata, JSON.stringify({
  source: input,
  method: 'median_IQR_robust_z_then_available_mean',
  note: 'Full-training transformation parameters. Cross-validation scores used fold-specific training-only parameters.',
  groups,
  removed,
  parameters,
  feature_count: featureOrder.length,
  feature_order: featureOrder,
}, null, 2));

console.log(JSON.stringify({ rows: rows.length, featureCount: featureOrder.length, totalColumns: outputHeader.length, output, metadata }, null, 2));
