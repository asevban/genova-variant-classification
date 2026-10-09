const fs=require('fs'),path=require('path');
const ROOT=__dirname,OUT=path.join(ROOT,'results');
function csv(t){const a=[];let r=[],c='',q=false;for(let i=0;i<t.length;i++){const x=t[i];if(q){if(x==='"'&&t[i+1]==='"'){c+='"';i++;}else if(x==='"')q=false;else c+=x;}else if(x==='"')q=true;else if(x===','){r.push(c);c='';}else if(x==='\n'){r.push(c.replace(/\r$/,''));a.push(r);r=[];c='';}else c+=x;}if(c.length||r.length){r.push(c);a.push(r);}return a;}
function read(f){const z=csv(fs.readFileSync(f,'utf8')),h=z[0];return z.slice(1).filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((x,i)=>[x,r[i]])));}
function mean(a){return a.reduce((s,v)=>s+v,0)/a.length;}
function met(z){let tn=0,fp=0,fn=0,tp=0;for(const r of z){if(r.actual===1&&r.predicted===1)tp++;else if(r.actual===0&&r.predicted===0)tn++;else if(r.actual===0)fp++;else fn++;}const sp=tn/(tn+fp)||0,re=tp/(tp+fn)||0,pr=tp/(tp+fp)||0,pr0=tn/(tn+fn)||0,f1=2*pr*re/(pr+re)||0,f10=2*pr0*sp/(pr0+sp)||0,pfp=100*(1-sp),ptp=20*re,pp=ptp/(ptp+pfp)||0,pf1=2*pp*re/(pp+re)||0,den=Math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return{tn,fp,fn,tp,accuracy:(tn+tp)/z.length,specificity:sp,recall:re,macro_f1:(f1+f10)/2,mcc:den?(tp*tn-fp*fn)/den:0,projected_f1:pf1,fp_per_100_benign:pfp};}
const all=read(path.join(OUT,'randomforest_319_hybrid_predictions.csv')).filter(r=>r.recipe==='ID3_CAT_RF_equal');
const groups=new Map();for(const r of all){const k=+r.row;if(!groups.has(k))groups.set(k,[]);groups.get(k).push({repeat:+r.repeat,actual:+r.actual,score:+r.score,threshold:+r.threshold});}
for(const [row,z] of groups)if(z.length!==5||new Set(z.map(x=>x.repeat)).size!==5)throw new Error(`Eksik tekrar: ${row}`);
const results=[];
function evaluate(label,kept){const decisions=[];for(const z of groups.values()){const use=z.filter(x=>kept.includes(x.repeat));decisions.push({actual:use[0].actual,predicted:mean(use.map(x=>x.score))>=mean(use.map(x=>x.threshold))?1:0});}results.push({configuration:label,models_used:kept.length,...met(decisions)});}
evaluate('all_5',[1,2,3,4,5]);for(let drop=1;drop<=5;drop++)evaluate(`drop_repeat_${drop}`,[1,2,3,4,5].filter(x=>x!==drop));
const h=Object.keys(results[0]);fs.writeFileSync(path.join(OUT,'cross_fitted_leave_one_model_out.csv'),[h.join(','),...results.map(r=>h.map(k=>r[k]).join(','))].join('\n'));
let md='# Cross-fitted Bagging — Bir Modeli Eksiltme Testi\n\nHer varyant için bulunan beş bağımsız cross-fitted olasılıktan sırasıyla biri çıkarılmış, kalan dört olasılık ve dört iç-CV eşiği ortalanmıştır. Amaç, başarıyı tek bir tekrar modelinin sürükleyip sürüklemediğini ölçmektir.\n\n|Yapı|Model sayısı|Proj. F1|MCC|Recall|Specificity|FP/100|TP|FP|FN|TN|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n';
for(const r of results)md+=`|${r.configuration}|${r.models_used}|${r.projected_f1.toFixed(4)}|${r.mcc.toFixed(4)}|${r.recall.toFixed(4)}|${r.specificity.toFixed(4)}|${r.fp_per_100_benign.toFixed(2)}|${r.tp}|${r.fp}|${r.fn}|${r.tn}|\n`;
const drops=results.slice(1),f1=drops.map(x=>x.projected_f1),mcc=drops.map(x=>x.mcc),fp=drops.map(x=>x.fp_per_100_benign);
md+=`\nDört modelli yapıların projeksiyon F1 aralığı **${Math.min(...f1).toFixed(4)}–${Math.max(...f1).toFixed(4)}**, MCC aralığı **${Math.min(...mcc).toFixed(4)}–${Math.max(...mcc).toFixed(4)}**, FP/100 aralığı **${Math.min(...fp).toFixed(2)}–${Math.max(...fp).toFixed(2)}** olmuştur.\n`;
fs.writeFileSync(path.join(OUT,'CROSS_FITTED_LEAVE_ONE_MODEL_OUT_RAPORU.md'),md,'utf8');console.log(md);
