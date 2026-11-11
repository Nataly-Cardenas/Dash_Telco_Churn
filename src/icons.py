"""Íconos de línea propios (24×24, trazo 2) que se pintan con el color del texto mediante una máscara CSS."""
from urllib.parse import quote
from dash import html

PATHS = {
    'signal': 'M4 20v-4 M9 20v-8 M14 20V8 M19 20V4',
    'chart': 'M3 3v18h18 M8 17v-6 M13 17V7 M18 17v-4',
    'flask': 'M9 3h6 M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3 M7.5 15h9',
    'trophy': 'M8 21h8 M12 17v4 M7 4h10v5a5 5 0 0 1-10 0z M17 6h3v2a3 3 0 0 1-3 3 M7 6H4v2a3 3 0 0 0 3 3',
    'sliders': 'M4 6h16 M4 12h16 M4 18h16 M14 4v4 M8 10v4 M16 16v4',
    'search': 'M11 4a7 7 0 1 0 0 14a7 7 0 1 0 0-14z M20 20l-4-4',
    'flag': 'M5 21V4 M5 4h12l-2 4 2 4H5',
    'users': 'M9 11a4 4 0 1 0 0-8a4 4 0 0 0 0 8z M2 21v-1a6 6 0 0 1 12 0v1 M16 3.5a4 4 0 0 1 0 7.5 M22 21v-1a6 6 0 0 0-4-5.6',
    'book': 'M4 5a2 2 0 0 1 2-2h14v16H6a2 2 0 0 0-2 2z M4 19V5 M8 7h8',
    'shield': 'M12 3l8 3v6c0 5-3.5 8-8 9c-4.5-1-8-4-8-9V6z M9 12l2 2 4-4',
    'trend': 'M3 17l6-6 4 4 8-8 M15 7h6v6',
    'flame': 'M12 3c3 4 6 6 6 10a6 6 0 0 1-12 0c0-3 2-4 3-7c1 2 2 3 3 3c0-2 0-4 0-6z',
    'network': 'M10 5a2 2 0 1 0 4 0a2 2 0 1 0-4 0 M3 19a2 2 0 1 0 4 0a2 2 0 1 0-4 0 M17 19a2 2 0 1 0 4 0a2 2 0 1 0-4 0 M12 7v4 M12 11l-6 6 M12 11l6 6',
    'table': 'M3 5h18v14H3z M3 10h18 M9 10v9',
    'target': 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8z M12 12h.01',
    'scale': 'M12 3v18 M7 21h10 M5 6h14 M2 14l3-8 3 8a3 3 0 0 1-6 0z M16 14l3-8 3 8a3 3 0 0 1-6 0z',
    'block': 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z M5.6 5.6l12.8 12.8',
    'tree': 'M12 3l6 8h-4l5 6H5l5-6H6z M12 17v4',
    'repeat': 'M17 2l4 4-4 4 M3 12v-2a4 4 0 0 1 4-4h14 M7 22l-4-4 4-4 M21 12v2a4 4 0 0 1-4 4H3',
    'sigma': 'M18 5V4H6l6 8-6 8h12v-1',
    'bulb': 'M9 18h6 M10 21h4 M12 3a6 6 0 0 0-4 10.5c.8.8 1 1.5 1 2.5h6c0-1 .2-1.7 1-2.5A6 6 0 0 0 12 3z',
    'card': 'M3 6h18v12H3z M3 10h18 M7 15h4',
    'wifi': 'M2 9a14 14 0 0 1 20 0 M5 12.5a10 10 0 0 1 14 0 M8.5 16a5 5 0 0 1 7 0 M12 19.5h.01',
    'download': 'M12 4v12 M7 11l5 5 5-5 M5 20h14',
    'mail': 'M3 6h18v12H3z M3 7l9 6 9-6',
    'code': 'M8 8l-4 4 4 4 M16 8l4 4-4 4 M14 4l-4 16',
    'file': 'M14 3H6v18h12V7z M14 3v4h4 M9 13h6 M9 17h6',
    'shuffle': 'M16 3h5v5 M4 20L21 3 M21 16v5h-5 M15 15l6 6 M4 4l5 5',
    'phone': 'M7 2h10v20H7z M11 18h2',
    'alert': 'M12 3l10 18H2z M12 10v4 M12 17h.01',
    'clock': 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z M12 7v5l3 2',
    'globe': 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z M3 12h18 M12 3a14 14 0 0 1 0 18 M12 3a14 14 0 0 0 0 18',
    'layers': 'M12 3l9 5-9 5-9-5z M3 13l9 5 9-5',
    'link': 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1 M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1',
    'gauge': 'M12 14l4-4 M3.5 18a9 9 0 1 1 17 0',
    'chat': 'M4 5h16v11H9l-5 4z M8 9h8 M8 12h5',
    'send': 'M4 12l16-8-6 16-3-7z M11 13l9-9',
    'close': 'M6 6l12 12 M18 6L6 18',
    'sun': 'M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8z M12 2v2 M12 20v2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4 M2 12h2 M20 12h2 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4',
    'moon': 'M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z',
    'user': 'M12 12a4 4 0 1 0 0-8a4 4 0 0 0 0 8z M5 21a7 7 0 0 1 14 0',
    'hourglass': 'M6 3h12 M6 21h12 M7 3c0 5 5 6 5 9s-5 4-5 9 M17 3c0 5-5 6-5 9s5 4 5 9',
    'reset': 'M4 4v6h6 M4.5 15a8 8 0 1 0 1.9-8.3L4 10',
    'cursor': 'M5 3l14 7-6 2-2 6z',
    'check': 'M5 12l5 5 9-10',
    'x': 'M6 6l12 12 M18 6L6 18',
    'arrow': 'M7 17L17 7 M8 7h9v9',
    'play': 'M7 4l12 8-12 8z',
    'linkedin': 'M4 4h16v16H4z M8 10v6 M8 7v.01 M12 16v-6 M12 12.5a2.5 2.5 0 0 1 5 0V16',
    'antenna': 'M12 13v8 M9 21h6 M12 13a2 2 0 1 0 0-4a2 2 0 1 0 0 4z M8.5 7.5a5 5 0 0 0 0 7 M15.5 7.5a5 5 0 0 1 0 7 M5.6 4.6a9 9 0 0 0 0 12.8 M18.4 4.6a9 9 0 0 1 0 12.8',
    'volume': 'M4 9h4l5-4v14l-5-4H4z M16 9a4 4 0 0 1 0 6 M18.5 6.5a8 8 0 0 1 0 11',
    'mute': 'M4 9h4l5-4v14l-5-4H4z M17 9l5 6 M22 9l-5 6',
    'eye': 'M2 12s4-7 10-7s10 7 10 7s-4 7-10 7S2 12 2 12z M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6z',
}


def url(name):
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' "
           f"stroke-linecap='round' stroke-linejoin='round'><path d='{PATHS[name]}'/></svg>")
    return f'url("data:image/svg+xml,{quote(svg)}")'


def icon(name, cls=''):
    u = url(name)
    return html.Span(className='ic ' + cls, style={'maskImage': u, 'WebkitMaskImage': u}, **{'aria-hidden': 'true'})
