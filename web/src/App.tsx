import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { Home } from './pages/Home';
import { Launchpad } from './pages/Launchpad';
import { DataDownload } from './pages/DataDownload';
import { Inspector } from './pages/Inspector';
import { useEffect } from 'react';

function App() {
  // Force dark mode
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  return (
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
  );
}

export default App;

