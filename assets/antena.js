/* Antena, la asistente de Señal perdida: un robot de telecomunicaciones con brillo neón.
   - Dibuja el personaje (SVG animado) en cada elemento .antena-bot de la página.
   - Saluda con voz femenina al abrir el chat (Web Speech API), en español o inglés según el idioma del dash.
   - De vez en cuando dice algo en un globo de diálogo junto al botón.
   El botón del altavoz la silencia y la elección se guarda en el navegador. */
(function () {
  'use strict';
  var GREET = {
    es: 'Hola, soy Antena, estoy aquí para resolver tus dudas. ¿En qué te puedo ayudar?',
    en: "Hi, I'm Antena, I'm here to answer your questions. How can I help you?"
  };
  var LINES = {
    es: ['¡Hola! Soy Antena', '¿Te ayudo con el churn?', '¡Analizando señales!', 'Pregúntame lo que quieras', 'Tengo buena cobertura de datos'],
    en: ["Hi! I'm Antena", 'Need help with churn?', 'Analyzing signals!', 'Ask me anything', 'Full data coverage here']
  };
  // Voces femeninas conocidas (Windows, Edge, Chrome, macOS y Android), de la más natural a la más común.
  var FEMALE = {
    es: [/dalia/i, /elvira/i, /helena/i, /sabina/i, /laura/i, /paloma/i, /salome|salomé/i, /paulina/i, /m[oó]nica/i, /luc[ií]a/i, /marisol/i, /google espa/i, /female|mujer/i],
    en: [/aria/i, /jenny/i, /michelle/i, /zira/i, /samantha/i, /karen/i, /google us english/i, /female/i]
  };
  var MALE = /\b(pablo|ra[uú]l|jorge|[aá]lvaro|diego|alonso|gerardo|guy|davis|david|mark|daniel|tony|enrique|carlos|juan|male)\b/i;
  var synth = window.speechSynthesis;
  var key = 'antena-muted';

  /* ---------- el personaje ---------- */
  var SVG = '' +
    '<svg class="bot" viewBox="0 0 120 150" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
    '<defs>' +
    '<linearGradient id="ab-body" x1="0" y1="0" x2="0.3" y2="1"><stop offset="0" stop-color="#8fd8f5"/><stop offset=".45" stop-color="#3f9ccb"/><stop offset="1" stop-color="#235f86"/></linearGradient>' +
    '<linearGradient id="ab-screen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d2238"/><stop offset="1" stop-color="#050c18"/></linearGradient>' +
    '<radialGradient id="ab-bulb" cx=".35" cy=".35" r=".7"><stop offset="0" stop-color="#ffd3bf"/><stop offset=".55" stop-color="#F06A45"/><stop offset="1" stop-color="#b8401f"/></radialGradient>' +
    '<radialGradient id="ab-ball" cx=".35" cy=".35" r=".7"><stop offset="0" stop-color="#e6f8ff"/><stop offset=".6" stop-color="#6cc3e8"/><stop offset="1" stop-color="#2E86AB"/></radialGradient>' +
    '</defs>' +
    '<ellipse class="bot-shadow" cx="60" cy="135" rx="25" ry="4"/>' +
    '<g class="bot-waves" fill="none" stroke-linecap="round">' +
    '<path class="w1" d="M51 14 a12 12 0 0 1 18 0"/><path class="w2" d="M45 9 a20 20 0 0 1 30 0"/><path class="w3" d="M39 4 a28 28 0 0 1 42 0"/>' +
    '</g>' +
    '<g class="bot-float">' +
    // antenas
    '<g class="bot-antennas" stroke="#a9bccd" stroke-width="3" stroke-linecap="round">' +
    '<line x1="60" y1="35" x2="60" y2="22"/><line x1="45" y1="37" x2="40" y2="27"/><line x1="75" y1="37" x2="80" y2="27"/><line x1="36" y1="42" x2="29" y2="34"/>' +
    '</g>' +
    '<circle class="bulb b0" cx="60" cy="20" r="4.5" fill="url(#ab-ball)"/>' +
    '<circle class="bulb b1" cx="39" cy="25" r="4" fill="url(#ab-bulb)"/><circle class="bulb b2" cx="81" cy="25" r="4" fill="url(#ab-bulb)"/>' +
    '<path d="M21 31 a8 8 0 0 1 12 -6 z" fill="#c9d6e2" stroke="#8aa0b5" stroke-width="1.2" transform="rotate(-20 27 30)"/>' +
    // brazo izquierdo (abajo)
    '<g class="arm-l"><path d="M31 82 q-11 9 -11 22" stroke="#a9bccd" stroke-width="5.5" fill="none" stroke-linecap="round"/>' +
    '<circle cx="20" cy="106" r="6" fill="url(#ab-body)" stroke="#9be3ff" stroke-width="1"/></g>' +
    // cuerpo y pantalla
    '<rect class="bot-body" x="28" y="34" width="64" height="74" rx="27" fill="url(#ab-body)" stroke="#bdeeff" stroke-width="1.6"/>' +
    '<path d="M34 54 q2 -14 16 -17" stroke="#ffffff" stroke-opacity=".35" stroke-width="3" fill="none" stroke-linecap="round"/>' +
    '<rect x="36" y="47" width="48" height="35" rx="11" fill="url(#ab-screen)" stroke="#6cc3e8" stroke-width="1.5"/>' +
    '<g class="bot-eyes" fill="#8ff6ff"><ellipse class="eye" cx="51" cy="62" rx="4.2" ry="5.2"/><ellipse class="eye" cx="69" cy="62" rx="4.2" ry="5.2"/></g>' +
    '<ellipse cx="44.5" cy="71" rx="3.6" ry="2.1" fill="#F06A45" opacity=".55"/><ellipse cx="75.5" cy="71" rx="3.6" ry="2.1" fill="#F06A45" opacity=".55"/>' +
    '<path class="mouth" d="M54 70 q6 5.5 12 0" stroke="#8ff6ff" stroke-width="2.4" fill="none" stroke-linecap="round"/>' +
    '<ellipse class="mouth-open" cx="60" cy="72" rx="4.5" ry="3.2" fill="#8ff6ff"/>' +
    // pecho y circuitos
    '<path d="M33 66 h4 v9 h4 M86 70 h-4 v8 h-3" stroke="#bdeeff" stroke-opacity=".7" stroke-width="1.3" fill="none"/>' +
    '<circle cx="41" cy="75" r="1.4" fill="#bdeeff"/><circle cx="79" cy="78" r="1.4" fill="#bdeeff"/>' +
    '<rect class="chest" x="50" y="89" width="20" height="5" rx="2.5" fill="#8ff6ff"/><circle class="led" cx="76" cy="91.5" r="2.2" fill="#F06A45"/>' +
    // piernas
    '<g fill="#a9bccd"><rect x="45" y="105" width="8" height="13" rx="2"/><rect x="67" y="105" width="8" height="13" rx="2"/></g>' +
    '<g fill="url(#ab-body)" stroke="#9be3ff" stroke-width="1"><rect x="40" y="116" width="17" height="9" rx="4.5"/><rect x="63" y="116" width="17" height="9" rx="4.5"/></g>' +
    // brazo derecho (saluda)
    '<g class="arm-r"><path d="M89 80 q12 -6 15 -22" stroke="#a9bccd" stroke-width="5.5" fill="none" stroke-linecap="round"/>' +
    '<g fill="url(#ab-body)" stroke="#9be3ff" stroke-width="1"><circle cx="105" cy="53" r="6.5"/>' +
    '<rect x="99.5" y="40" width="3.4" height="9" rx="1.7"/><rect x="103.6" y="38.5" width="3.4" height="10" rx="1.7"/><rect x="107.7" y="40.5" width="3.4" height="9" rx="1.7"/><rect x="110.5" y="47" width="3.2" height="7" rx="1.6" transform="rotate(30 112 50)"/></g></g>' +
    '</g></svg>';

  // Cada dibujo usa ids propios: si dos SVG comparten ids de degradado y el primero está oculto, Chrome no pinta el segundo.
  var count = 0;
  function draw(el) { if (!el.dataset.drawn) { var n = 'ab' + (++count) + '-'; el.innerHTML = SVG.replace(/ab-/g, n); el.dataset.drawn = '1'; } }
  function scan() { document.querySelectorAll('.antena-bot').forEach(draw); paintMute(); }

  /* ---------- voz ---------- */
  function muted() { try { return localStorage.getItem(key) === '1'; } catch (e) { return false; } }
  function setMuted(m) { try { localStorage.setItem(key, m ? '1' : '0'); } catch (e) {} paintMute(); }
  function paintMute() { var c = document.getElementById('chat'); if (c) c.classList.toggle('muted', muted()); }
  function language() { return (document.documentElement.lang || 'es').slice(0, 2) === 'en' ? 'en' : 'es'; }

  function pickVoice(lang) {
    var pool = synth.getVoices().filter(function (v) { return (v.lang || '').toLowerCase().indexOf(lang) === 0; });
    for (var i = 0; i < FEMALE[lang].length; i++) {
      for (var j = 0; j < pool.length; j++) if (FEMALE[lang][i].test(pool[j].name)) return pool[j];
    }
    var notMale = pool.filter(function (v) { return !MALE.test(v.name); });
    return notMale[0] || pool[0] || null;
  }

  function speaking(on) { document.querySelectorAll('.antena-bot').forEach(function (f) { f.classList.toggle('speaking', on); }); }

  function greet(retry) {
    if (!synth || muted()) return;
    if (!synth.getVoices().length && !retry) {          // las voces cargan de forma asíncrona la primera vez
      var done = false, go = function () { if (!done) { done = true; greet(true); } };
      synth.addEventListener('voiceschanged', go, { once: true }); setTimeout(go, 700); return;
    }
    synth.cancel();
    var lang = language(), u = new SpeechSynthesisUtterance(GREET[lang]), v = pickVoice(lang);
    if (v) { u.voice = v; u.lang = v.lang; } else { u.lang = lang === 'en' ? 'en-US' : 'es-MX'; }
    u.rate = 1; u.pitch = 1.12;
    u.onstart = function () { speaking(true); }; u.onend = u.onerror = function () { speaking(false); };
    synth.speak(u);
  }
  function stop() { if (synth) synth.cancel(); speaking(false); }

  /* ---------- globo de diálogo junto al botón ---------- */
  var line = 0;
  function bubble() {
    var b = document.querySelector('.bot-bubble'), c = document.getElementById('chat');
    if (!b || (c && c.classList.contains('open'))) return;
    var lines = LINES[language()]; b.textContent = lines[line++ % lines.length];
    b.classList.add('show'); document.querySelectorAll('.chat-fab .antena-bot').forEach(function (f) { f.classList.add('waving'); });
    setTimeout(function () { b.classList.remove('show'); document.querySelectorAll('.chat-fab .antena-bot').forEach(function (f) { f.classList.remove('waving'); }); }, 4200);
  }
  setTimeout(bubble, 2500); setInterval(bubble, 24000);

  document.addEventListener('click', function (e) {
    var t = e.target.closest ? e.target : e.target.parentElement;
    if (t.closest('#chat-fab')) {
      var b = document.querySelector('.bot-bubble'); if (b) b.classList.remove('show');
      // El panel se abre con un callback del navegador; se espera a que cambie la clase.
      setTimeout(function () { var c = document.getElementById('chat'); if (c && c.classList.contains('open')) greet(); else stop(); }, 120);
    }
    if (t.closest('#chat-close')) stop();
    if (t.closest('#chat-voice')) { var m = !muted(); setMuted(m); if (m) stop(); else greet(); }
  });
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener('DOMContentLoaded', scan);
  if (synth) synth.getVoices();
})();
