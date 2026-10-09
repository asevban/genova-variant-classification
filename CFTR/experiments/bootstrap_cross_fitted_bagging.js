const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const INPUT = path.join(ROOT, 'results', 'cross_fitted_bagging_variant_predictions.csv');
const OUT_CSV = path.join(ROOT, 'results', 'cross_fitted_bagging_cluster_bootstrap.csv');
const OUT_MD = path.join(ROOT, 'results', 'CROSS_FITTED_BAGGING_BOOTSTRAP_RAPORU.md');
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
function quantile(sorted,p){const x=(sorted.length-1)*p,l=Math.floor(x),d=x-l;return sorted[l]+(sorted[Math.min(l+1,sorted.length-1)]-sorted[l])*d;}
function metrics(records){
  let tn=0,fp=0,fn=0,tp=0;
  for(const r of records){if(r.actual===1&&r.pred===1)tp++;else if(r.actual===0&&r.pred===0)tn++;else if(r.actual===0)fp++;else fn++;}
  const recall=tp/(tp+fn)||0, specificity=tn/(tn+fp)||0, precision=tp/(tp+fp)||0;
  const observedF1=2*precision*recall/(precision+recall)||0;
  const projectedTP=20*recall, projectedFP=100*(1-specificity), projectedPrecision=projectedTP/(projectedTP+projectedFP)||0;
  const projectedF1=2*projectedPrecision*recall/(projectedPrecision+recall)||0;
  const den=Math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); const mcc=den?(tp*tn-fp*fn)/den:0;
  return {observed_f1:observedF1,projected_f1:projectedF1,mcc,recall,specificity,fp_per_100_benign:100*(1-specificity),tn,fp,fn,tp};
}

const raw=table(INPUT);
const modes=[
  ['probability_average','pred_probability_average'],
  ['majority_vote','pred_majority_vote'],
  ['conservative_4of5','pred_conservative_4of5']
];
const output=[];
for(const [mode,column] of modes){
  const records=raw.map(r=>({actual:Number(r.actual),pred:Number(r[column])}));
  const benign=records.filter(r=>r.actual===0), pathogenic=records.filter(r=>r.actual===1);
  if(benign.length!==21||pathogenic.length!==90)throw new Error(`Beklenmeyen sınıf sayısı: ${benign.length}/${pathogenic.length}`);
  const point=metrics(records), names=['observed_f1','projected_f1','mcc','recall','specificity','fp_per_100_benign'];
  const dist=Object.fromEntries(names.map(n=>[n,[]])); const random=rng(20260902 + modes.findIndex(x=>x[0]===mode));
  for(let b=0;b<B;b++){
    const sampled=[];
    for(let i=0;i<benign.length;i++)sampled.push(benign[Math.floor(random()*benign.length)]);
    for(let i=0;i<pathogenic.length;i++)sampled.push(pathogenic[Math.floor(random()*pathogenic.length)]);
    const m=metrics(sampled); for(const n of names)dist[n].push(m[n]);
  }
  for(const n of names){const d=dist[n].sort((a,b)=>a-b);output.push({mode,metric:n,point:point[n],ci_low:quantile(d,.025),ci_high:quantile(d,.975),bootstrap_repetitions:B});}
}
const header=Object.keys(output[0]);
fs.writeFileSync(OUT_CSV,[header.join(','),...output.map(r=>header.map(h=>r[h]).join(','))].join('\n'));
let md='# Cross-fitted Bagging — Variant_ID Kümeli Bootstrap\n\n';
md+='Her varyant tek bağımsız küme kabul edilmiştir. Benign ve patojenik sınıflar ayrı ayrı yeniden örneklenerek eğitim setindeki `21 benign + 90 patojenik` sayısı korunmuştur. Hesaplama 100.000 bootstrap tekrarıyla yapılmıştır.\n\n';
md+='|Karar yöntemi|Metrik|Nokta tahmini|%95 güven aralığı|\n|---|---|---:|---:|\n';
for(const r of output)md+=`|${r.mode}|${r.metric}|${r.point.toFixed(4)}|${r.ci_low.toFixed(4)} – ${r.ci_high.toFixed(4)}|\n`;
md+='\n## Yorum\n\nGüven aralığı, aynı büyüklükte yeni varyant örneklemlerinde skorun ne kadar oynayabileceğini gösterir. Final test dağılımına uyarlanmış F1 için nokta tahmini tek başına yeterli değildir; özellikle alt sınır ve `FP/100 benign` üst sınırı birlikte değerlendirilmelidir. Bu analiz final test etiketlerini kullanmaz.\n';
fs.writeFileSync(OUT_MD,md,'utf8');
console.log(md);
