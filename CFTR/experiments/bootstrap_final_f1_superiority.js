const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const INPUT = path.join(ROOT, 'results', 'randomforest_319_hybrid_predictions.csv');
const DATA = path.join(ROOT, 'YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv');
const OUT = path.join(ROOT, 'results');
const B = 100000;

function parseCSV(text) {
  const rows=[]; let row=[], cell='', quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(quoted){if(c==='"'&&text[i+1]==='"'){cell+='"';i++;}else if(c==='"')quoted=false;else cell+=c;}
    else if(c==='"')quoted=true; else if(c===','){row.push(cell);cell='';}
    else if(c==='\n'){row.push(cell.replace(/\r$/,''));rows.push(row);row=[];cell='';} else cell+=c;
  }
  if(cell.length||row.length){row.push(cell);rows.push(row);} return rows;
}
function table(file){const r=parseCSV(fs.readFileSync(file,'utf8'));const h=r[0];return r.slice(1).filter(x=>x.length===h.length).map(x=>Object.fromEntries(h.map((k,i)=>[k,x[i]])));}
function rng(seed){return function(){let t=seed+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function metrics(records){let tn=0,fp=0,fn=0,tp=0;for(const r of records){if(r.actual===1&&r.pred===1)tp++;else if(r.actual===0&&r.pred===0)tn++;else if(r.actual===0)fp++;else fn++;}
  const recall=tp/(tp+fn)||0,specificity=tn/(tn+fp)||0,precision=tp/(tp+fp)||0;
  const observedF1=2*precision*recall/(precision+recall)||0;
  const projectedTP=20*recall,projectedFP=100*(1-specificity),projectedPrecision=projectedTP/(projectedTP+projectedFP)||0;
  const projectedF1=2*projectedPrecision*recall/(projectedPrecision+recall)||0;
  const den=Math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));const mcc=den?(tp*tn-fp*fn)/den:0;
  return {observed_f1:observedF1,projected_f1:projectedF1,mcc,recall,specificity,fp_per_100_benign:100*(1-specificity)};
}
function quantile(sorted,p){const x=(sorted.length-1)*p,l=Math.floor(x),d=x-l;return sorted[l]+(sorted[Math.min(l+1,sorted.length-1)]-sorted[l])*d;}

const variants=table(DATA);const variantId=variants.map(r=>r.Variant_ID);
const pred=table(INPUT).filter(r=>r.recipe==='ID3_CAT_RF_equal'||r.recipe==='baseline_ID3_CAT_ADA');
const byModel={main:new Map(),backup:new Map()};
for(const r of pred){const model=r.recipe==='ID3_CAT_RF_equal'?'main':'backup';const row=Number(r.row);if(!byModel[model].has(row))byModel[model].set(row,[]);byModel[model].get(row).push({actual:Number(r.actual),pred:Number(r.predicted)});}
const actualByRow=new Map();for(const [row,recs] of byModel.main)actualByRow.set(row,recs[0].actual);
const benign=[...actualByRow].filter(([,a])=>a===0).map(([r])=>r),pathogenic=[...actualByRow].filter(([,a])=>a===1).map(([r])=>r);
if(benign.length!==21||pathogenic.length!==90)throw new Error(`Beklenmeyen sınıf sayısı: ${benign.length}/${pathogenic.length}`);
for(const row of [...benign,...pathogenic])if(byModel.main.get(row).length!==5||byModel.backup.get(row).length!==5)throw new Error(`Eksik tekrar: ${row}`);

const pointMain=metrics([...byModel.main.values()].flat()),pointBackup=metrics([...byModel.backup.values()].flat());
const names=['observed_f1','projected_f1','mcc','recall','specificity','fp_per_100_benign'];const dist=Object.fromEntries(names.map(n=>[n,[]]));
const random=rng(20260831);
for(let b=0;b<B;b++){
  const sampled=[];for(let i=0;i<benign.length;i++)sampled.push(benign[Math.floor(random()*benign.length)]);for(let i=0;i<pathogenic.length;i++)sampled.push(pathogenic[Math.floor(random()*pathogenic.length)]);
  const main=[],backup=[];for(const row of sampled){main.push(...byModel.main.get(row));backup.push(...byModel.backup.get(row));}
  const mm=metrics(main),mb=metrics(backup);for(const n of names)dist[n].push(mm[n]-mb[n]);
}
const rows=[];
for(const n of names){const d=dist[n].sort((a,b)=>a-b);rows.push({metric:n,main:pointMain[n],backup:pointBackup[n],delta:pointMain[n]-pointBackup[n],ci_low:quantile(d,.025),ci_high:quantile(d,.975),prob_main_better:d.filter(x=>x>0).length/B,bootstrap_repetitions:B});}
const header=Object.keys(rows[0]);fs.writeFileSync(path.join(OUT,'final_f1_superiority_cluster_bootstrap.csv'),[header.join(','),...rows.map(r=>header.map(h=>r[h]).join(','))].join('\n'));
let md='# Nihai Model F1 Üstünlüğü — 100.000 Tekrarlı Paired Cluster Bootstrap\n\nHer `Variant_ID` bir küme olarak örneklenmiş, aynı varyanta ait beş tekrar tahmini birlikte tutulmuştur. Benign ve patojenik varyantlar kendi sınıfları içinde yeniden örneklenerek sınıf oranı korunmuştur.\n\n|Metrik|Ana model|Yedek model|Fark|%95 GA|Ana modelin üstünlük olasılığı|\n|---|---:|---:|---:|---:|---:|\n';
for(const r of rows)md+=`|${r.metric}|${r.main.toFixed(4)}|${r.backup.toFixed(4)}|${r.delta.toFixed(4)}|${r.ci_low.toFixed(4)} – ${r.ci_high.toFixed(4)}|%${(100*r.prob_main_better).toFixed(1)}|\n`;
md+='\nBir farkın %95 düzeyinde kesin üstünlük sayılması için güven aralığının tamamının sıfırın üzerinde olması gerekir.\n';fs.writeFileSync(path.join(OUT,'FINAL_F1_USTUNLUGU_BOOTSTRAP_RAPORU.md'),md);console.log(md);
