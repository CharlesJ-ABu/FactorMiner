import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { Home } from './pages/Home';
import { Launchpad } from './pages/Launchpad';
import { DataDownload } from './pages/DataDownload';
import { Inspector } from './pages/Inspector';
import { useEffect } from 'react';
import { I18nProvider } from './i18n';

function App() {
  // Force dark mode
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <I18nProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Home />} />
            <Route path="launchpad" element={<Launchpad />} />
            <Route path="data" element={<DataDownload />} />
            <Route path="inspector" element={<Inspector />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </I18nProvider>
  );
}

export default App;
