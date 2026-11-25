import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { Home } from './pages/Home';
import { lazy, useEffect } from 'react';
import { I18nProvider } from './i18n';

const Launchpad = lazy(() => import('./pages/Launchpad').then(page => ({ default: page.Launchpad })));
const DataDownload = lazy(() => import('./pages/DataDownload').then(page => ({ default: page.DataDownload })));
const Inspector = lazy(() => import('./pages/Inspector').then(page => ({ default: page.Inspector })));

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
