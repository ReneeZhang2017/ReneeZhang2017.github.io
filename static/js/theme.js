(() => {
 const button = document.querySelector('.theme');
 function setTheme(dark) {
  document.body.classList.toggle('dark', dark);
  button.textContent = dark ? '☀' : '☾';
  button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
  try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch {}
 }
 let dark = false;
 try { dark = localStorage.getItem('theme') === 'dark'; } catch {}
 setTheme(dark);
 button.addEventListener('click', () => setTheme(!document.body.classList.contains('dark')));
})();
