let entries = [];
let dataPath = '';
let idx = 0;
let saveTimer = null;

const $ = (id) => document.getElementById(id);

async function init() {
  const state = await window.api.getState();
  entries = state.entries || [];
  dataPath = state.dataPath || '';
  if (!entries.length) {
    $('meta').textContent = 'No entries loaded.';
    return;
  }
  const firstUnapproved = entries.findIndex(e =>
    e.gt_approved === null || e.gt_approved === undefined
  );
  idx = firstUnapproved >= 0 ? firstUnapproved : 0;
  buildScoreButtons();
  render();
}

function buildScoreButtons() {
  const box = $('scorebtns');
  box.innerHTML = '';
  for (let i = 0; i <= 10; i++) {
    const v = (i / 10).toFixed(1);
    const btn = document.createElement('button');
    btn.className = 'scorebtn';
    btn.textContent = v;
    btn.dataset.value = v;
    btn.addEventListener('click', () => setApproved(parseFloat(v)));
    box.appendChild(btn);
  }
}

function flashSaved(msg = '✓ saved') {
  $('save-status').textContent = msg;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => { $('save-status').textContent = ''; }, 1500);
}

function render() {
  if (!entries.length) return;
  const e = entries[idx];
  const approvedCount = entries.filter(x =>
    x.gt_approved !== null && x.gt_approved !== undefined
  ).length;

  // Badges — V5 model vs GT decision
  const v5IsDropped = (e.v5_dropped === undefined) ? e.dropped : e.v5_dropped;
  const v5Badge = v5IsDropped
    ? '<span style="background:#a1260d;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;margin-right:4px;font-size:10px;">V5:DROPPED</span>'
    : '<span style="background:#16825d;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;margin-right:4px;font-size:10px;">V5:KEPT</span>';
  const gtBadge = (e.gt_action === 'drop')
    ? '<span style="background:#a1260d;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;margin-right:8px;font-size:10px;">GT:DROP</span>'
    : '<span style="background:#0e639c;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;margin-right:8px;font-size:10px;">GT:COMPRESS</span>';

  // Agreement marker
  const agree = (v5IsDropped === (e.gt_action === 'drop'))
    ? '<span style="background:#16825d;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;margin-right:8px;font-size:10px;">✓ AGREE</span>'
    : '<span style="background:#cc8c00;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;margin-right:8px;font-size:10px;">⚠ DISAGREE</span>';

  const finishTag = e.v5_finish_reason && e.v5_finish_reason !== 'stop'
    ? ` · <span style="color:#f48771;">finish:${e.v5_finish_reason}</span>` : '';

  $('meta').innerHTML =
    `${v5Badge}${gtBadge}${agree}<b>${e.gt_label || ''}</b> · ${e.sample_id} · seg ${e.seg_id} · ` +
    `${e.level} · ${e.repo || '?'}${finishTag}`;
  $('progress').textContent =
    `[${idx + 1} / ${entries.length}]  approved: ${approvedCount}`;

  // User intent (truncated to 600 chars for display)
  let intentText = e.user_intent || '(none)';
  if (intentText.length > 600) intentText = intentText.slice(0, 600) + ' [...]';
  $('intent').textContent = intentText;

  // Stats — V5
  const useV5 = e.v5_compressed !== undefined || e.v5_dropped !== undefined;
  const v5Body = useV5 ? e.v5_compressed : e.compressed;
  const v5Dropped = useV5 ? e.v5_dropped : e.dropped;
  const v5Chars = useV5 ? (e.v5_chars || 0) : (e.dropped ? 0 : e.seg_compressed_chars);
  const v5Ratio = useV5 ? (e.v5_ratio || 0) : (e.dropped ? 0 : e.seg_ratio);

  const gtChars = e.gt_compressed ? e.gt_compressed.length : 0;
  const gtRatio = e.seg_original_chars
    ? (gtChars / e.seg_original_chars) : 0;

  $('orig-stats').textContent = `${e.seg_original_chars} chars`;
  $('teacher-stats').textContent = v5Dropped
    ? '[DROPPED]'
    : `${v5Chars} chars (${Math.round(v5Ratio * 100)}%)`;
  $('gt-stats').textContent = (e.gt_action === 'drop')
    ? '[DROP]'
    : `${gtChars} chars (${Math.round(gtRatio * 100)}%)`;

  // Panes
  $('original').textContent = e.original;
  if (v5Dropped || !v5Body) {
    $('compressed').innerHTML =
      '<div class="dropped">[DROPPED by V5]<br>This SEG is absent from the V5 model output.</div>';
  } else {
    $('compressed').textContent = v5Body;
  }

  // GT editor
  const gtTa = $('gt-text');
  if (e.gt_action === 'drop') {
    gtTa.value = e.gt_compressed || '';
    gtTa.classList.add('dropped-mode');
    gtTa.placeholder = '[DROP — no text. Switch action to "compress" to edit.]';
  } else {
    gtTa.value = e.gt_compressed || '';
    gtTa.classList.remove('dropped-mode');
    gtTa.placeholder = 'Edit GT compression here. Auto-saves on blur.';
  }

  // Rationale
  $('rationale').value = e.gt_rationale || '';

  // Action buttons
  $('act-compress').classList.toggle('active', e.gt_action === 'compress');
  $('act-drop').classList.toggle('active', e.gt_action === 'drop');

  // Score buttons
  document.querySelectorAll('.scorebtn').forEach(btn => {
    btn.classList.toggle('active', parseFloat(btn.dataset.value) === e.gt_approved);
  });
  $('curscore').textContent = (e.gt_approved !== null && e.gt_approved !== undefined)
    ? `Approved: ${Number(e.gt_approved).toFixed(1)}`
    : 'Not approved yet';

  $('prev').disabled = idx === 0;
  $('next').disabled = idx === entries.length - 1;

  $('original').scrollTop = 0;
  $('compressed').scrollTop = 0;
  gtTa.scrollTop = 0;
}

async function saveGt(fields) {
  const e = entries[idx];
  Object.assign(e, fields);
  // Recompute local derived
  if ('gt_compressed' in fields) {
    e.gt_chars = fields.gt_compressed ? fields.gt_compressed.length : 0;
    e.gt_ratio = e.seg_original_chars
      ? Math.round((e.gt_chars / e.seg_original_chars) * 1000) / 1000
      : 0;
  }
  await window.api.setGt(e.entry_id, fields);
  flashSaved();
  // Re-render stats only (not the whole pane) so the textarea cursor isn't lost
  const useV5 = e.v5_compressed !== undefined || e.v5_dropped !== undefined;
  const mDropped = useV5 ? e.v5_dropped : e.dropped;
  const mChars = useV5 ? (e.v5_chars || 0) : (e.dropped ? 0 : e.seg_compressed_chars);
  const mRatio = useV5 ? (e.v5_ratio || 0) : (e.dropped ? 0 : e.seg_ratio);
  $('teacher-stats').textContent = mDropped
    ? '[DROPPED]'
    : `${mChars} chars (${Math.round(mRatio * 100)}%)`;
  $('gt-stats').textContent = (e.gt_action === 'drop')
    ? '[DROP]'
    : `${e.gt_chars} chars (${Math.round(e.gt_ratio * 100)}%)`;
}

async function setApproved(score) {
  const e = entries[idx];
  e.gt_approved = score;
  await window.api.setGt(e.entry_id, { gt_approved: score });
  flashSaved('✓ approved');
  render();
}

async function clearApproved() {
  const e = entries[idx];
  e.gt_approved = null;
  await window.api.setGt(e.entry_id, { gt_approved: null });
  flashSaved('cleared');
  render();
}

function prev() { if (idx > 0) { idx--; render(); } }
function next() { if (idx < entries.length - 1) { idx++; render(); } }
function nextUnapproved() {
  for (let i = idx + 1; i < entries.length; i++) {
    if (entries[i].gt_approved === null || entries[i].gt_approved === undefined) {
      idx = i; render(); return;
    }
  }
  for (let i = 0; i < idx; i++) {
    if (entries[i].gt_approved === null || entries[i].gt_approved === undefined) {
      idx = i; render(); return;
    }
  }
}

// Wire up
$('prev').addEventListener('click', prev);
$('next').addEventListener('click', next);
$('next-unapproved').addEventListener('click', nextUnapproved);
$('clear').addEventListener('click', clearApproved);

$('pick').addEventListener('click', async () => {
  const s = await window.api.pickDataFile();
  if (!s) return;
  entries = s.entries; dataPath = s.dataPath;
  idx = 0;
  render();
});

// GT text auto-save on blur
$('gt-text').addEventListener('blur', () => {
  const newText = $('gt-text').value;
  const e = entries[idx];
  if (newText !== e.gt_compressed) {
    saveGt({ gt_compressed: newText });
  }
});

// Rationale auto-save on blur
$('rationale').addEventListener('blur', () => {
  const newText = $('rationale').value;
  const e = entries[idx];
  if (newText !== e.gt_rationale) {
    saveGt({ gt_rationale: newText });
  }
});

// Action toggle
$('act-compress').addEventListener('click', () => {
  saveGt({ gt_action: 'compress' });
  render();
});
$('act-drop').addEventListener('click', () => {
  saveGt({ gt_action: 'drop', gt_compressed: null });
  render();
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  // Don't intercept when typing in editor / rationale
  const inEditor = e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT';
  if (inEditor) {
    // Ctrl+Enter to save & next
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      e.target.blur();
      next();
    }
    return;
  }
  if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
  else if (e.key === 'u' || e.key === 'U') { e.preventDefault(); nextUnapproved(); }
  else if (e.key === 'c' || e.key === 'C') { e.preventDefault(); clearApproved(); }
  else if (/^[0-9]$/.test(e.key)) {
    e.preventDefault();
    const n = parseInt(e.key, 10);
    const score = (n === 0 && e.shiftKey) ? 1.0 : n / 10;
    setApproved(score);
  }
});

init();
