const fs=require('fs'),path=require('path');
const ROOT=__dirname,OUT=path.join(ROOT,'results');
function csv(t){const a=[];let r=[],c='',q=false;for(let i=0;i<t.length;i++){const x=t[i];if(q){if(x==='"'&&t[i+1]==='"'){c+='"';i++;}else if(x==='"')q=false;else c+=x;}else if(x==='"')q=true;else if(x===','){r.push(c);c='';}else if(x==='\n'){r.push(c.replace(/\r$/,''));a.push(r);r=[];c='';}else c+=x;}if(c.length||r.length){r.push(c);a.push(r);}return a;}
function read(file){const z=csv(fs.readFileSync(file,'utf8')),h=z[0];return z.slice(1).filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((x,i)=>[x,r[i]])));}
function esc(v){const s=String(v??'');return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function mean(a){return a.reduce((s,v)=>s+v,0)/a.length;}function sd(a){const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1||1));}
const data=read(path.join(ROOT,'YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv'));
const hybrid=read(path.join(OUT,'randomforest_319_hybrid_predictions.csv'));
const main=hybrid.filter(r=>r.recipe==='ID3_CAT_RF_equal'),backup=hybrid.filter(r=>r.recipe==='baseline_ID3_CAT_ADA');
const id3=read(path.join(OUT,'id3_benign_weighted_322_vs_320_predictions.csv')).filter(r=>r.key==='id3_scenario1d_319');
const cat=read(path.join(OUT,'catboost_benign_weighted_finalists_predictions.csv')).filter(r=>r.key==='scenario1_robust_groups_no_cat6');
const rf=read(path.join(OUT,'randomforest_319_nested_predictions.csv'));
function group(rows){const m=new Map();for(const r of rows){const k=Number(r.row);if(!m.has(k))m.set(k,[]);m.get(k).push(r);}return m;}
const maps={main:group(main),backup:group(backup),id3:group(id3),cat:group(cat),rf:group(rf)};const rows=[];
for(let row=0;row<data.length;row++){
 const a=Number(maps.main.get(row)[0].actual),mr=maps.main.get(row),br=maps.backup.get(row),ir=maps.id3.get(row),cr=maps.cat.get(row),rr=maps.rf.get(row);
 const scores=mr.map(x=>Number(x.score)),mp=mr.map(x=>Number(x.predicted)),bp=br.map(x=>Number(x.predicted)),ip=ir.map(x=>Number(x.predicted)),cp=cr.map(x=>Number(x.predicted)),rp=rr.map(x=>Number(x.predicted));
 const err=mp.filter(x=>x!==a).length,miss=data[row].Variant_ID;
 const missingCount=Object.entries(data[row]).filter(([k,v])=>k!=='Variant_ID'&&k!=='Label'&&(v===''||v==null)).length;
 rows.push({row,Variant_ID:miss,actual:a,main_errors_5:err,main_positive_votes_5:mp.reduce((s,v)=>s+v,0),main_score_mean:mean(scores),main_score_sd:sd(scores),backup_errors_5:bp.filter(x=>x!==a).length,id3_errors_5:ip.filter(x=>x!==a).length,catboost_errors_5:cp.filter(x=>x!==a).length,rf_errors_5:rp.filter(x=>x!==a).length,component_disagreement_repeats:mp.map((_,i)=>new Set([ip[i],cp[i],rp[i]]).size>1?1:0).reduce((s,v)=>s+v,0),missing_feature_count:missingCount,flag:err===5?(a===0?'CONSISTENT_FP':'CONSISTENT_FN'):err>=3?'MAJORITY_ERROR':err>0?'INTERMITTENT_ERROR':'CONSISTENT_CORRECT'});
}
rows.sort((a,b)=>b.main_errors_5-a.main_errors_5||b.component_disagreement_repeats-a.component_disagreement_repeats||a.row-b.row);
const head=Object.keys(rows[0]);fs.writeFileSync(path.join(OUT,'variant_error_audit.csv'),[head.join(','),...rows.map(r=>head.map(h=>esc(r[h])).join(','))].join('\n'));
const counts={};for(const r of rows)counts[r.flag]=(counts[r.flag]||0)+1;const fp=rows.filter(r=>r.flag==='CONSISTENT_FP'),fn=rows.filter(r=>r.flag==='CONSISTENT_FN'),maj=rows.filter(r=>r.flag==='MAJORITY_ERROR');
let md='# Varyant Bazlı Hata ve Veri Kalitesi Denetimi\n\nHer Variant_ID için ana modelin beş tekrardaki dış-fold tahminleri birleştirildi. Bu analiz etiket değiştirmez veya satır silmez; yalnız manuel inceleme adaylarını belirler.\n\n## Özet\n\n';
for(const [k,v] of Object.entries(counts))md+=`- ${k}: ${v}\n`;md+=`\n- Beş tekrarda sürekli FP: ${fp.length}\n- Beş tekrarda sürekli FN: ${fn.length}\n- En az 3/5 tekrar yanlış: ${maj.length+fp.length+fn.length}\n\n## Sürekli yanlış sınıflandırılan varyantlar\n\n|Variant_ID|Gerçek|Hata/5|Ort. skor|Skor SS|Eksik özellik|Bileşen anlaşmazlığı/5|\n|---|---:|---:|---:|---:|---:|---:|\n`;
for(const r of rows.filter(x=>x.main_errors_5===5))md+=`|${r.Variant_ID}|${r.actual}|${r.main_errors_5}|${r.main_score_mean.toFixed(4)}|${r.main_score_sd.toFixed(4)}|${r.missing_feature_count}|${r.component_disagreement_repeats}|\n`;
md+='\nBu kayıtlar otomatik olarak aykırı değer veya yanlış etiket kabul edilemez. Kaynak etiket, varyant gösterimi ve annotation değerleri bağımsız biyolojik kaynaktan doğrulanmadan veri değişikliği yapılmamalıdır.\n';fs.writeFileSync(path.join(OUT,'VARYANT_HATA_VE_VERI_KALITESI_DENETIMI.md'),md);console.log(md);
