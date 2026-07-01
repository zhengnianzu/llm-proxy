(function() {
  var STORAGE_KEY = 'llm-proxy-theme';

  function getPreferred() {
    var saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function apply(theme) {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem(STORAGE_KEY, theme);
    var icons = document.querySelectorAll('.theme-icon');
    icons.forEach(function(el) {
      el.textContent = theme === 'dark' ? '☀' : '☾';
    });
    var labels = document.querySelectorAll('.theme-label');
    labels.forEach(function(el) {
      el.textContent = theme === 'dark' ? '亮色模式' : '暗色模式';
    });
  }

  apply(getPreferred());

  window.toggleTheme = function() {
    var current = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
    apply(current === 'dark' ? 'light' : 'dark');
  };

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
    if (!localStorage.getItem(STORAGE_KEY)) {
      apply(e.matches ? 'dark' : 'light');
    }
  });
})();
