/* Señal Perdida — animaciones de la portada y contadores.
   Dash vuelve a montar el contenido al navegar, así que todo se inicializa
   cuando los elementos aparecen en el DOM. */
(function () {
  'use strict';
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Red de clientes alrededor del celular ---------- */
  function initNetwork(canvas) {
    if (canvas.dataset.ready) return;
    canvas.dataset.ready = '1';
    var ctx = canvas.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = 0, H = 0, hub = { x: 0, y: 0 };
    var nodes = [], pulses = [], mouse = { x: -999, y: -999 };
    var lost = 0, saved = 0;
    var lostEl = document.getElementById('cover-lost');
    var savedEl = document.getElementById('cover-saved');

    function hubPos() {
      var phone = document.getElementById('phone');
      var r = canvas.getBoundingClientRect();
      if (phone && phone.offsetParent) {
        var p = phone.getBoundingClientRect();
        hub.x = p.left + p.width / 2 - r.left; hub.y = p.top + p.height / 2 - r.top;
      } else { hub.x = W * 0.72; hub.y = H * 0.5; }
    }
    function resize() {
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      hubPos();
    }
    function spawn(n) {
      var a = Math.random() * Math.PI * 2;
      var radius = 120 + Math.random() * Math.max(W, H) * 0.55;
      n.a = a; n.r = radius; n.speed = (0.0006 + Math.random() * 0.0012) * (Math.random() < 0.5 ? -1 : 1);
      n.size = 1.6 + Math.random() * 2.4; n.state = 'on'; n.t = 0; n.alpha = 0; n.drift = 0;
      return n;
    }
    function populate() {
      var count = Math.round(Math.min(150, Math.max(60, W * H / 9000)));
      nodes = [];
      for (var i = 0; i < count; i++) nodes.push(spawn({}));
    }
    resize(); populate();
    window.addEventListener('resize', function () { resize(); });
    canvas.parentElement.addEventListener('mousemove', function (e) {
      var r = canvas.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
    });
    canvas.parentElement.addEventListener('mouseleave', function () { mouse.x = mouse.y = -999; });
    // en pantallas táctiles el dedo hace de cursor
    function touch(e) { var t = e.touches[0]; if (!t) return; var r = canvas.getBoundingClientRect(); mouse.x = t.clientX - r.left; mouse.y = t.clientY - r.top; }
    canvas.parentElement.addEventListener('touchstart', touch, { passive: true });
    canvas.parentElement.addEventListener('touchmove', touch, { passive: true });
    canvas.parentElement.addEventListener('touchend', function () { mouse.x = mouse.y = -999; });

    var last = 0, nextChurn = 0;
    function frame(ts) {
      if (!document.body.contains(canvas)) return;              // la portada se desmontó
      requestAnimationFrame(frame);
      var dt = Math.min(50, ts - last || 16); last = ts;
      if (ts % 4 < 1) hubPos();
      ctx.clearRect(0, 0, W, H);
      var light = document.documentElement.dataset.theme === 'light';

      // ondas de señal que salen del celular
      if (!pulses.length || pulses[pulses.length - 1].r > 70) pulses.push({ r: 20 });
      pulses.forEach(function (p) { p.r += dt * 0.09; });
      pulses = pulses.filter(function (p) { return p.r < Math.max(W, H) * 0.7; });
      pulses.forEach(function (p) {
        var a = Math.max(0, 0.28 - p.r / (Math.max(W, H) * 2.4));
        ctx.beginPath(); ctx.arc(hub.x, hub.y, p.r, 0, Math.PI * 2);
        ctx.strokeStyle = (light ? 'rgba(46,134,171,' : 'rgba(79,163,199,') + a + ')'; ctx.lineWidth = 1.2; ctx.stroke();
      });

      // ~1 de cada 4 clientes termina abandonando: se dispara una fuga periódica
      if (ts > nextChurn && !reduced) {
        var candidates = nodes.filter(function (n) { return n.state === 'on' && n.alpha > 0.9; });
        if (candidates.length) { var c = candidates[Math.floor(Math.random() * candidates.length)]; c.state = 'risk'; c.t = 0; }
        nextChurn = ts + 650 + Math.random() * 900;
      }

      nodes.forEach(function (n) {
        n.a += n.speed * dt * (reduced ? 0 : 1);
        n.t += dt;
        if (n.state === 'on' || n.state === 'saved') n.alpha = Math.min(1, n.alpha + dt * 0.002);
        if (n.state === 'risk' && n.t > 1500) { n.state = 'gone'; n.t = 0; lost++; if (lostEl) lostEl.textContent = lost.toLocaleString('en-US'); }
        if (n.state === 'gone') { n.drift += dt * 0.12; n.alpha -= dt * 0.0011; if (n.alpha <= 0) spawn(n); }
        if (n.state === 'saved' && n.t > 2200) n.state = 'on';

        var x = hub.x + Math.cos(n.a) * (n.r + n.drift);
        var y = hub.y + Math.sin(n.a) * (n.r + n.drift) * 0.62;
        n.x = x; n.y = y;

        // el cursor retiene a los clientes en riesgo
        if ((n.state === 'risk' || n.state === 'gone') && n.drift < 160) {
          var dx = x - mouse.x, dy = y - mouse.y;
          if (dx * dx + dy * dy < 70 * 70) {
            n.state = 'saved'; n.t = 0; n.drift = 0; n.alpha = 1; saved++;
            if (savedEl) savedEl.textContent = saved.toLocaleString('en-US');
          }
        }

        var color = n.state === 'risk' || n.state === 'gone' ? (light ? '228,87,46' : '240,106,69') : n.state === 'saved' ? (light ? '46,134,171' : '79,163,199') : (light ? '120,140,160' : '150,172,196');
        var a = Math.max(0, n.alpha);
        // enlace al celular: se corta cuando el cliente se va
        if (n.state !== 'gone') {
          var flicker = n.state === 'risk' ? (Math.sin(n.t / 60) > 0 ? 0.5 : 0.08) : 0.11;
          ctx.beginPath(); ctx.moveTo(hub.x, hub.y); ctx.lineTo(x, y);
          ctx.strokeStyle = 'rgba(' + color + ',' + flicker * a + ')'; ctx.lineWidth = n.state === 'saved' ? 1.4 : 0.7; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(x, y, n.size + (n.state === 'risk' ? 1.5 : 0), 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + color + ',' + a + ')';
        ctx.shadowColor = 'rgba(' + color + ',.9)'; ctx.shadowBlur = n.state === 'on' ? 6 : 16;
        ctx.fill(); ctx.shadowBlur = 0;
      });
    }
    requestAnimationFrame(frame);
  }

  /* ---------- Celular: clientes reales del test con su riesgo ---------- */
  function initPhone(phone) {
    if (phone.dataset.ready) return;
    phone.dataset.ready = '1';
    var customers = [];
    try { customers = JSON.parse(phone.getAttribute('data-customers') || '[]'); } catch (e) { customers = []; }
    if (!customers.length) return;
    var text = {};
    try { text = JSON.parse(phone.getAttribute('data-text') || '{}'); } catch (e) { text = {}; }
    var i = 0;
    var ring = phone.querySelector('.ring');
    function q(sel) { return phone.querySelector(sel); }
    function show() {
      if (!document.body.contains(phone)) return;
      var c = customers[i % customers.length]; i++;
      var screen = q('.phone-body'); screen.classList.remove('swap'); void screen.offsetWidth; screen.classList.add('swap');
      q('.pc-id').textContent = c.id;
      q('.pc-contract').textContent = c.contract;
      q('.pc-internet').textContent = c.internet;
      q('.pc-payment').textContent = c.payment;
      q('.pc-tenure').textContent = c.tenure + ' ' + (text.months || 'meses');
      q('.pc-monthly').textContent = '$' + c.monthly.toFixed(2);
      var ul = q('.pc-flags'); ul.innerHTML = '';
      (c.flags || []).forEach(function (f) { var li = document.createElement('li'); li.textContent = f; li.className = f.charAt(0) === '▲' ? 'up' : 'down'; ul.appendChild(li); });
      var pct = Math.round(c.prob * 100);
      q('.ring-pct').textContent = pct + '%';
      var level = c.prob >= 0.6 ? 'alto' : c.prob >= 0.35 ? 'medio' : 'bajo';
      phone.dataset.level = level;
      q('.pc-level').textContent = (text.risk || {})[level] || level;
      q('.pc-truth').textContent = c.churn ? text.churned : text.stayed;
      phone.dataset.truth = c.churn ? 'churn' : 'stay';
      ring.style.setProperty('--p', pct);
      var bars = phone.querySelectorAll('.sig b'), lit = Math.max(1, Math.round((1 - c.prob) * 4));
      bars.forEach(function (b, k) { b.classList.toggle('lit', k < lit); });
      var toast = q('.toast');
      q('.toast-text').textContent = (text.toast || {})[level] || '';
      toast.classList.remove('show'); void toast.offsetWidth; toast.classList.add('show');
      setTimeout(show, 3600);
    }
    show();

    // inclinación 3D con el cursor
    var stage = phone.parentElement;
    stage.addEventListener('mousemove', function (e) {
      var r = stage.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width - 0.5, py = (e.clientY - r.top) / r.height - 0.5;
      phone.style.transform = 'rotateY(' + (px * 22) + 'deg) rotateX(' + (-py * 18) + 'deg)';
    });
    stage.addEventListener('mouseleave', function () { phone.style.transform = ''; });
  }

  /* ---------- Contadores animados ---------- */
  function initCounter(el) {
    if (el.dataset.ready === el.dataset.count) return;   // Dash puede reutilizar el nodo con otro valor
    el.dataset.ready = el.dataset.count;
    var target = parseFloat(el.dataset.count), dec = parseInt(el.dataset.decimals || '0', 10);
    var suffix = el.dataset.suffix || '', prefix = el.dataset.prefix || '';
    var fmt = function (v) { return prefix + v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix; };
    if (reduced) { el.textContent = fmt(target); return; }
    var start = null, dur = 1600;
    function step(ts) {
      if (!start) start = ts;
      var k = Math.min(1, (ts - start) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(target * e);
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function scan() {
    var c = document.getElementById('net-canvas'); if (c) initNetwork(c);
    var p = document.getElementById('phone'); if (p) initPhone(p);
    document.querySelectorAll('[data-count]').forEach(initCounter);
  }
  new MutationObserver(function () { scan(); }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-count'] });
  document.addEventListener('DOMContentLoaded', scan);
})();
