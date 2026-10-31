/* Antena, la asistente de Señal perdida: saluda con voz femenina al abrir el chat.
   Usa la síntesis de voz del navegador (Web Speech API), en español o inglés según el idioma del dash.
   El botón del altavoz la silencia y la elección se guarda en el navegador. */
(function () {
  'use strict';
  var GREET = {
    es: 'Hola, soy Antena, estoy aquí para resolver tus dudas. ¿En qué te puedo ayudar?',
    en: "Hi, I'm Antena, I'm here to answer your questions. How can I help you?"
  };
  // Voces femeninas conocidas (Windows, Edge, Chrome, macOS y Android), de la más natural a la más común.
  var FEMALE = {
    es: [/dalia/i, /elvira/i, /helena/i, /sabina/i, /laura/i, /paloma/i, /salome|salomé/i, /paulina/i, /m[oó]nica/i, /luc[ií]a/i, /marisol/i, /google espa/i, /female|mujer/i],
    en: [/aria/i, /jenny/i, /michelle/i, /zira/i, /samantha/i, /karen/i, /google us english/i, /female/i]
  };
  var MALE = /\b(pablo|ra[uú]l|jorge|[aá]lvaro|diego|alonso|gerardo|jorge|guy|davis|david|mark|daniel|tony|enrique|carlos|juan|male)\b/i;
  var synth = window.speechSynthesis;
  var key = 'antena-muted';

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

  function speaking(on) { document.querySelectorAll('.antena-face').forEach(function (f) { f.classList.toggle('speaking', on); }); }

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

  document.addEventListener('click', function (e) {
    var t = e.target.closest ? e.target : e.target.parentElement;
    if (t.closest('#chat-fab')) {
      // El panel se abre con un callback del navegador; se espera a que cambie la clase.
      setTimeout(function () { var c = document.getElementById('chat'); if (c && c.classList.contains('open')) greet(); else stop(); }, 120);
    }
    if (t.closest('#chat-close')) stop();
    if (t.closest('#chat-voice')) { var m = !muted(); setMuted(m); if (m) stop(); else greet(); }
  });
  new MutationObserver(paintMute).observe(document.documentElement, { childList: true, subtree: true });
  if (synth) synth.getVoices();
})();
