"""The HTML dashboard served at GET /stats when a browser asks for it (Accept:
text/html). Programmatic callers (curl, the hosted meter) still get the JSON
snapshot — handle_stats content-negotiates on Accept.

Self-contained: inline CSS + vanilla JS, no CDN (the proxy is a local server and
must not reach out). The page fetches the same /stats endpoint as JSON every few
seconds and renders it, accumulating a live token-savings series for the chart
while it's open (the proxy keeps cumulative totals, not time-series, so the
"over time" view is built client-side from the poll). Styled to match
paritok.com's palette.
"""

STATS_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paritok · live stats</title>
<style>
  :root{
    --bg:#07080c; --bg-soft:#0b0d14; --card:#0f121b; --elevated:#141826;
    --line:#1e2333; --brand:#6d5efc; --brand-light:#8b7dff; --brand-glow:#a78bfa;
    --accent:#22d3ee; --green:#34d399; --ink:#eef1f8; --ink-muted:#9aa3b8; --ink-faint:#6b7488;
    --neg:#f87171;
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{
    background:var(--bg); color:var(--ink); min-height:100vh;
    font-family:var(--font-sans,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif);
    -webkit-font-smoothing:antialiased;
    background-image:radial-gradient(60rem 40rem at 85% -10%, rgba(109,94,252,.10), transparent 60%),
                     radial-gradient(50rem 40rem at 5% 0%, rgba(34,211,238,.06), transparent 55%);
    background-attachment:fixed;
  }
  .mono{font-family:var(--font-mono,ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace);
        font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1;}
  .wrap{max-width:1120px; margin:0 auto; padding:28px 22px 56px}

  header{display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:26px}
  .brand{display:flex; align-items:center; gap:11px}
  .logo{width:30px;height:30px;border-radius:8px;
    background:linear-gradient(135deg,var(--brand),var(--accent));
    box-shadow:0 0 0 1px rgba(255,255,255,.06), 0 8px 24px -8px rgba(109,94,252,.7);
    position:relative}
  .logo::after{content:"";position:absolute;inset:8px;border-radius:3px;background:var(--bg);opacity:.55}
  .brand h1{font-size:16px;font-weight:650;letter-spacing:-.01em;margin:0}
  .brand .sub{font-size:12px;color:var(--ink-muted);margin-top:1px}
  .status{display:flex;align-items:center;gap:14px;font-size:12.5px;color:var(--ink-muted)}
  .live{display:inline-flex;align-items:center;gap:7px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(52,211,153,.6);
    animation:pulse 2s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.55)}70%{box-shadow:0 0 0 7px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}

  .banner{display:none; margin-bottom:22px; padding:13px 16px; border:1px solid var(--line);
    border-radius:12px; background:var(--bg-soft); color:var(--ink-muted); font-size:13.5px}
  .banner.show{display:block}
  .banner b{color:var(--ink)}
  .banner code{background:#000;color:var(--brand-light);padding:2px 7px;border-radius:6px;
    font-family:var(--font-mono,ui-monospace,monospace);font-size:12.5px}

  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
  @media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}}
  @media(max-width:460px){.kpis{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 17px;
    animation:fade .5s cubic-bezier(.16,1,.3,1) both}
  @keyframes fade{0%{opacity:0;transform:translateY(10px)}100%{opacity:1;transform:none}}
  .card .k-label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-faint)}
  .card .k-val{font-size:30px;font-weight:600;letter-spacing:-.02em;margin-top:8px;line-height:1}
  .card .k-sub{font-size:12px;color:var(--ink-muted);margin-top:8px}
  .accent-green{color:var(--green)} .accent-brand{color:var(--brand-light)} .accent-cyan{color:var(--accent)}
  .neg{color:var(--neg)}

  .grid2{display:grid;grid-template-columns:1.55fr 1fr;gap:14px;margin-bottom:16px}
  @media(max-width:820px){.grid2{grid-template-columns:1fr}}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:17px 18px}
  .panel h2{font-size:13px;font-weight:600;margin:0 0 2px;letter-spacing:-.01em}
  .panel .h-sub{font-size:11.5px;color:var(--ink-faint);margin-bottom:14px}

  /* compression part-to-whole bar */
  .cbar{height:22px;border-radius:7px;background:var(--elevated);overflow:hidden;display:flex;
    border:1px solid var(--line)}
  .cbar .kept{background:linear-gradient(90deg,var(--brand),var(--brand-light));height:100%;min-width:2px;
    border-radius:6px 0 0 6px;transition:width .5s cubic-bezier(.16,1,.3,1)}
  .clabels{display:flex;justify-content:space-between;margin-top:9px;font-size:12px;color:var(--ink-muted)}
  .clabels b{color:var(--ink)}

  .splits{display:flex;gap:10px;margin-top:16px}
  .split{flex:1;background:var(--bg-soft);border:1px solid var(--line);border-radius:10px;padding:11px 12px}
  .split .s-l{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-faint)}
  .split .s-v{font-size:19px;font-weight:600;margin-top:5px}
  .s-unit{font-size:12px;font-weight:400;color:var(--ink-faint);margin-left:5px;letter-spacing:0}

  /* chart */
  .chart-wrap{position:relative}
  svg{display:block;width:100%;height:190px;overflow:visible}
  .grid-line{stroke:var(--line);stroke-width:1;opacity:.55}
  .area{fill:url(#g);opacity:.9}
  .line{fill:none;stroke:var(--green);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  .cross{stroke:var(--ink-faint);stroke-width:1;stroke-dasharray:3 3;opacity:0}
  .cdot{fill:var(--green);stroke:var(--bg);stroke-width:2;opacity:0}
  .y-tick{fill:var(--ink-faint);font-size:10px}
  .tip{position:absolute;pointer-events:none;opacity:0;transform:translate(-50%,-120%);
    background:var(--elevated);border:1px solid var(--line);border-radius:8px;padding:6px 9px;
    font-size:12px;white-space:nowrap;transition:opacity .1s;box-shadow:0 8px 24px -10px #000}
  .tip .tv{font-weight:600}
  .chart-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    color:var(--ink-faint);font-size:12.5px}

  .mini{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:16px}
  @media(max-width:820px){.mini{grid-template-columns:repeat(2,1fr)}}
  @media(max-width:460px){.mini{grid-template-columns:1fr}}
  .mini .split{margin:0}
  .s-sub{font-size:10.5px;color:var(--ink-faint);margin-top:3px;letter-spacing:0}

  /* recent compressions — original → compressed before/after, one per page */
  .samples{margin-top:26px}
  .samples-head{display:flex;align-items:center;justify-content:space-between;gap:12px;
    margin-bottom:12px;flex-wrap:wrap}
  .samples-head h2{font-size:13px;font-weight:600;margin:0;letter-spacing:-.01em}
  .samples-head .h-sub{font-size:11.5px;color:var(--ink-faint);margin-top:2px}
  .pager{display:flex;align-items:center;gap:8px}
  .pg-btn{width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;
    background:var(--card);color:var(--ink-muted);border:1px solid var(--line);border-radius:9px;
    font-size:17px;line-height:1;cursor:pointer;transition:.15s;padding:0}
  .pg-btn:hover:not(:disabled){color:var(--ink);border-color:var(--brand);background:var(--elevated)}
  .pg-btn:disabled{opacity:.35;cursor:default}
  .pg-label{font-size:12.5px;color:var(--ink-muted);min-width:52px;text-align:center}
  .sample{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 17px;
    animation:fade .4s cubic-bezier(.16,1,.3,1) both}
  .sample .s-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:13px}
  .pill{font-size:10.5px;letter-spacing:.03em;color:var(--ink-muted);background:var(--bg-soft);
    border:1px solid var(--line);border-radius:999px;padding:3px 9px}
  .save-pill{font-size:12px;font-weight:600;color:var(--green);background:rgba(52,211,153,.10);
    border:1px solid rgba(52,211,153,.25);border-radius:999px;padding:3px 9px}
  .sample .toks{font-size:12.5px;color:var(--ink-muted);margin-left:auto}
  .sample .toks b{color:var(--ink)}
  .ba{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:720px){.ba{grid-template-columns:1fr}}
  .ba .c-l{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-faint);
    margin-bottom:6px;display:flex;align-items:center;gap:6px}
  .ba .c-l .arrow{color:var(--brand-light);font-weight:700}
  pre.code{margin:0;background:var(--bg-soft);border:1px solid var(--line);border-radius:9px;
    padding:12px 13px;font-family:var(--font-mono,ui-monospace,SFMono-Regular,Menlo,Consolas,monospace);
    font-size:12px;line-height:1.55;color:var(--ink-muted);max-height:360px;overflow:auto;
    white-space:pre-wrap;word-break:break-word;tab-size:2}
  pre.code.compressed{color:var(--ink);border-color:rgba(109,94,252,.30)}
  .samples-empty{color:var(--ink-faint);font-size:12.5px;padding:4px 0}

  footer{margin-top:24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;
    font-size:12px;color:var(--ink-faint)}
  footer a{color:var(--ink-muted);text-decoration:none;border-bottom:1px solid var(--line)}
  footer a:hover{color:var(--ink)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo"></div>
      <div><h1>Paritok</h1><div class="sub">Local proxy · live compression stats</div></div>
    </div>
    <div class="status">
      <span class="live"><span class="dot"></span> Live</span>
      <span id="uptime" class="mono">—</span>
    </div>
  </header>

  <div id="banner" class="banner">
    <b>No traffic yet.</b> Point your agent at this proxy
    (<code id="proxyUrl">http://127.0.0.1:8080</code>) and savings will appear here.
  </div>

  <div class="kpis">
    <div class="card"><div class="k-label">Tokens saved</div>
      <div id="kSaved" class="k-val mono accent-green">0</div>
      <div id="kSavedSub" class="k-sub">across 0 requests</div></div>
    <div class="card"><div class="k-label">Est. cost saved</div>
      <div id="kCost" class="k-val mono accent-cyan">$0.00</div>
      <div class="k-sub">based on the model's input rate</div></div>
    <div class="card"><div class="k-label">Compression</div>
      <div id="kRatio" class="k-val mono accent-brand">—</div>
      <div id="kRatioSub" class="k-sub">of intercepted input kept</div></div>
    <div class="card"><div class="k-label">Requests</div>
      <div id="kReq" class="k-val mono">0</div>
      <div id="kReqSub" class="k-sub">processed</div></div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Tokens saved over time</h2>
      <div class="h-sub">Live — accumulated while this page is open</div>
      <div class="chart-wrap">
        <svg id="chart" viewBox="0 0 800 190" preserveAspectRatio="none">
          <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#34d399" stop-opacity=".34"/>
            <stop offset="1" stop-color="#34d399" stop-opacity="0"/>
          </linearGradient></defs>
          <g id="grid"></g>
          <path id="area" class="area" d=""/>
          <path id="line" class="line" d=""/>
          <line id="cross" class="cross" y1="0" y2="190"/>
          <circle id="cdot" class="cdot" r="4"/>
        </svg>
        <div id="chartEmpty" class="chart-empty">waiting for the first data point…</div>
        <div id="tip" class="tip"><span id="tipV" class="tv mono"></span></div>
      </div>
    </div>

    <div class="panel">
      <h2>Where the savings come from</h2>
      <div class="h-sub">Intercepted input, original → forwarded</div>
      <div class="cbar"><div id="keptBar" class="kept" style="width:0%"></div></div>
      <div class="clabels"><span>original <b id="cOrig" class="mono">0</b></span>
        <span><b id="cComp" class="mono">0</b> forwarded</span></div>
      <div class="splits">
        <div class="split"><div class="s-l">File / output compression</div>
          <div class="s-v"><span id="fileSaved" class="mono accent-green">0</span><span class="s-unit">tok saved</span></div></div>
        <div class="split"><div class="s-l">Tool-schema filtering</div>
          <div class="s-v"><span id="toolSaved" class="mono accent-brand">0</span><span class="s-unit">tok saved</span></div></div>
      </div>
    </div>
  </div>

  <div class="mini">
    <div class="split"><div class="s-l">Tools filtered</div>
      <div class="s-v"><span id="mTools" class="mono">0</span><span class="s-unit">tools dropped</span></div></div>
    <div class="split"><div class="s-l">Context expansions</div><div id="mExp" class="s-v mono">0</div></div>
    <div class="split"><div class="s-l">Edits recovered</div><div id="mEdit" class="s-v mono">0</div></div>
    <div class="split"><div class="s-l">Passthrough (not compressed)</div>
      <div class="s-v"><span id="mSkip" class="mono">0</span><span class="s-unit">tok skipped</span></div>
      <div id="mSkipSub" class="s-sub"></div></div>
  </div>

  <div class="samples">
    <div class="samples-head">
      <div><h2>Recent compressions</h2>
        <div class="h-sub">What the model actually received — original → compressed, newest first</div></div>
      <div id="pager" class="pager">
        <button id="pgPrev" class="pg-btn" title="Newer">‹</button>
        <span id="pgLabel" class="pg-label mono">0 / 0</span>
        <button id="pgNext" class="pg-btn" title="Older">›</button>
      </div>
    </div>
    <div id="sampleView"></div>
    <div id="samplesEmpty" class="samples-empty">No compressions captured yet.</div>
  </div>

  <footer>
    <span>Auto-refreshing every 3s · self-hosted, nothing leaves your box</span>
    <a href="https://www.paritok.com" target="_blank" rel="noopener">paritok.com ↗</a>
  </footer>
</div>

<script>
(function(){
  var W=800,H=190;
  var series=[]; // {t, v}
  var $=function(id){return document.getElementById(id)};
  document.getElementById("proxyUrl").textContent = location.origin || "http://127.0.0.1:8080";

  function fmt(n){
    n = Math.round(Number(n)||0);
    var neg = n<0; n=Math.abs(n);
    return (neg?"-":"") + n.toLocaleString("en-US");
  }
  function fmtDur(s){
    s=Math.max(0,Math.floor(s||0));
    var d=Math.floor(s/86400); s-=d*86400;
    var h=Math.floor(s/3600); s-=h*3600;
    var m=Math.floor(s/60); var ss=s-m*60;
    if(d) return d+"d "+h+"h";
    if(h) return h+"h "+m+"m";
    if(m) return m+"m "+ss+"s";
    return ss+"s";
  }

  function draw(){
    var grid=$("grid"); grid.innerHTML="";
    var empty=$("chartEmpty");
    if(series.length<2){ empty.style.display="flex"; $("area").setAttribute("d",""); $("line").setAttribute("d",""); return; }
    empty.style.display="none";
    var vals=series.map(function(p){return p.v});
    var min=Math.min.apply(null,vals), max=Math.max.apply(null,vals);
    if(max===min){ max=min+1; }
    var pad=(max-min)*0.12; min-=pad; max+=pad;
    // gridlines + y ticks (3)
    for(var i=0;i<=3;i++){
      var gy=(H/3)*i;
      var ln=document.createElementNS("http://www.w3.org/2000/svg","line");
      ln.setAttribute("x1",0);ln.setAttribute("x2",W);ln.setAttribute("y1",gy);ln.setAttribute("y2",gy);
      ln.setAttribute("class","grid-line"); grid.appendChild(ln);
      var val=max-((max-min)/3)*i;
      var tx=document.createElementNS("http://www.w3.org/2000/svg","text");
      tx.setAttribute("x",4);tx.setAttribute("y",gy-4<8?12:gy-4);tx.setAttribute("class","y-tick");
      tx.textContent=fmt(val); grid.appendChild(tx);
    }
    var n=series.length;
    var X=function(i){return n<2?0:(i/(n-1))*W};
    var Y=function(v){return H-((v-min)/(max-min))*H};
    var d="M"+X(0)+","+Y(series[0].v);
    for(var i=1;i<n;i++) d+="L"+X(i)+","+Y(series[i].v);
    $("line").setAttribute("d",d);
    $("area").setAttribute("d",d+"L"+W+","+H+"L0,"+H+"Z");
    window.__X=X; window.__Y=Y;
  }

  // hover crosshair + tooltip
  var svg=$("chart"), tip=$("tip"), cross=$("cross"), cdot=$("cdot");
  svg.addEventListener("mousemove", function(e){
    if(series.length<2) return;
    var r=svg.getBoundingClientRect();
    var fx=(e.clientX-r.left)/r.width;
    var idx=Math.round(fx*(series.length-1));
    idx=Math.max(0,Math.min(series.length-1,idx));
    var px=window.__X(idx), py=window.__Y(series[idx].v);
    var sx=(px/W)*r.width, sy=(py/H)*r.height;
    cross.setAttribute("x1",px);cross.setAttribute("x2",px);cross.style.opacity=1;
    cdot.setAttribute("cx",px);cdot.setAttribute("cy",py);cdot.style.opacity=1;
    tip.style.left=sx+"px"; tip.style.top=sy+"px"; tip.style.opacity=1;
    $("tipV").textContent=fmt(series[idx].v)+" saved";
  });
  svg.addEventListener("mouseleave", function(){cross.style.opacity=0;cdot.style.opacity=0;tip.style.opacity=0;});

  function el(tag,cls,txt){var e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e;}

  // One original→compressed pair per page, fetched one at a time from
  // /stats?sample=N (N=0 is newest). Local-only: the pairs hold real file content
  // and tool schemas, so they never leave this box.
  var sIdx=0, sTotal=0, sLoaded=false, _rk="";
  function renderSample(resp){
    var total=resp.samples_total||0, s=resp.sample;
    // Skip the rebuild when nothing changed, so the newest-page auto-refresh every
    // tick doesn't nuke the DOM (and the user's scroll position in the panes).
    var key=(resp.sample_index||0)+"/"+total+"/"+
      (s?(s.original_tokens+"-"+s.compressed_tokens+"-"+(s.compressed||"").length):"none");
    if(key===_rk) return;
    _rk=key;
    sTotal=total;
    var view=$("sampleView"), empty=$("samplesEmpty"), pager=$("pager");
    if(!total || !s){
      view.innerHTML=""; empty.style.display="block"; pager.style.visibility="hidden";
      $("pgLabel").textContent="0 / 0"; sLoaded=false; return;
    }
    empty.style.display="none"; pager.style.visibility="visible"; sLoaded=true;
    sIdx=resp.sample_index||0;
    view.innerHTML="";
    var card=el("div","sample"), head=el("div","s-head");
    head.appendChild(el("span","pill", s.source||"content"));
    if(s.model) head.appendChild(el("span","pill", s.model));
    head.appendChild(el("span","save-pill", "−"+Math.round((1-(s.kept_ratio||0))*100)+"%"));
    var toks=el("span","toks mono");
    toks.appendChild(el("b",null,fmt(s.original_tokens)));
    toks.appendChild(document.createTextNode(" → "));
    toks.appendChild(el("b",null,fmt(s.compressed_tokens)));
    toks.appendChild(document.createTextNode(" tok"));
    head.appendChild(toks); card.appendChild(head);
    var ba=el("div","ba");
    var c1=el("div"), l1=el("div","c-l"); l1.textContent="original"; c1.appendChild(l1);
    c1.appendChild(el("pre","code", s.original||""));
    var c2=el("div"), l2=el("div","c-l");
    l2.appendChild(el("span","arrow","→")); l2.appendChild(document.createTextNode("compressed"));
    c2.appendChild(l2); c2.appendChild(el("pre","code compressed", s.compressed||""));
    ba.appendChild(c1); ba.appendChild(c2); card.appendChild(ba);
    view.appendChild(card);
    $("pgLabel").textContent=(sIdx+1)+" / "+total;
    $("pgPrev").disabled=(sIdx<=0);          // ‹ newer
    $("pgNext").disabled=(sIdx>=total-1);    // › older
  }
  function fetchSample(idx){
    fetch("/stats?sample="+idx,{headers:{Accept:"application/json"},cache:"no-store"})
      .then(function(r){return r.json()}).then(renderSample).catch(function(){});
  }
  $("pgPrev").addEventListener("click",function(){ if(sIdx>0) fetchSample(sIdx-1); });
  $("pgNext").addEventListener("click",function(){ if(sIdx<sTotal-1) fetchSample(sIdx+1); });
  function syncSamples(count){
    count=count||0;
    if(!count){ renderSample({samples_total:0,sample:null}); return; }
    // Not loaded yet, or pinned to the newest page → always pull the current
    // newest. The count saturates at the window cap, so it can't be used to detect
    // a fresh pair once the window is full; fetchSample re-renders idempotently and
    // renderSample skips the DOM rebuild when the pair is unchanged (scroll-safe).
    if(!sLoaded || sIdx===0){ fetchSample(0); return; }
    // On an older page: keep the view stable while reading, just refresh the total.
    sTotal=count;
    $("pgLabel").textContent=(sIdx+1)+" / "+count;
    $("pgNext").disabled=(sIdx>=count-1);
  }

  function apply(d){
    var req=d.total_requests||0;
    $("banner").className = req>0 ? "banner" : "banner show";
    var saved=d.tokens_saved||0;
    var kv=$("kSaved"); kv.textContent=fmt(saved);
    kv.className="k-val mono "+(saved<0?"neg":"accent-green");
    $("kSavedSub").textContent="across "+fmt(req)+" request"+(req===1?"":"s");
    $("kCost").textContent=d.estimated_cost_saved_usd||"$0.00";
    var ratio=d.compression_ratio||0;
    $("kRatio").textContent=req? Math.round(ratio*100)+"%" : "—";
    $("kRatioSub").textContent=req? "of intercepted input kept · "+Math.round((1-ratio)*100)+"% dropped" : "of intercepted input kept";
    $("kReq").textContent=fmt(req);
    $("uptime").textContent="up "+fmtDur(d.uptime_seconds);

    var orig=d.input_tokens_original||0, comp=d.input_tokens_compressed||0;
    $("cOrig").textContent=fmt(orig); $("cComp").textContent=fmt(comp);
    var pct = orig? Math.max(0,Math.min(100,(comp/orig)*100)) : 0;
    $("keptBar").style.width=pct+"%";
    $("fileSaved").textContent=fmt(d.file_compression_saved);
    $("toolSaved").textContent=fmt(d.tool_filter_saved);
    $("mTools").textContent=fmt(d.tools_filtered);
    $("mExp").textContent=fmt(d.expansions);
    $("mEdit").textContent=fmt(d.edits_recovered);
    $("mSkip").textContent=fmt(d.tokens_skipped);
    var byr=d.skipped_by_reason||{}, rk=Object.keys(byr);
    rk.sort(function(a,b){return byr[b]-byr[a];});
    $("mSkipSub").textContent = rk.length ? rk.slice(0,2).map(function(k){return k+" "+fmt(byr[k]);}).join(" · ") : "";
    syncSamples(d.compression_samples_count);

    // accumulate the live series (only push when it changes or on first point)
    var last=series.length?series[series.length-1].v:null;
    if(last===null || saved!==last){
      series.push({t:Date.now(), v:saved});
      if(series.length>240) series.shift();
      draw();
    }
  }

  function tick(){
    fetch("/stats",{headers:{Accept:"application/json"},cache:"no-store"})
      .then(function(r){return r.json()}).then(apply).catch(function(){});
  }
  tick(); setInterval(tick,3000);
})();
</script>
</body>
</html>"""
