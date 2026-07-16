import { Outlet, Link, useLocation } from 'react-router-dom';
import { Rocket, FileSearch, Database, LayoutDashboard, Languages } from 'lucide-react';
import { useI18n } from '../i18n';

export function MainLayout() {
  const location = useLocation();
  const { language, setLanguage, t } = useI18n();

  const navItems = [
    { name: t('nav.dashboard'), path: '/', icon: LayoutDashboard },
    { name: t('nav.launchpad'), path: '/launchpad', icon: Rocket },
    { name: t('nav.data'), path: '/data', icon: Database },
    { name: t('nav.inspector'), path: '/inspector', icon: FileSearch },
  ];

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Top Navbar */}
      <header className="h-16 flex items-center px-6 border-b border-border bg-card shadow-sm z-10">
        <div className="flex items-center gap-2 text-primary font-bold text-xl tracking-tight mr-10">
          <Rocket className="h-6 w-6 text-blue-500" />
          FactorMiner V4
        </div>
        
        <nav className="flex-1 flex items-center space-x-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium transition-colors ${
                  isActive 
                    ? 'bg-primary/10 text-primary' 
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? 'text-primary' : ''}`} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2 ml-auto">
          <div className="flex items-center gap-1 px-2 py-1.5 bg-secondary/50 rounded-full border border-border">
            <Languages className="h-3.5 w-3.5 text-muted-foreground" />
            <select value={language} onChange={(event) => setLanguage(event.target.value as typeof language)} aria-label="Language" className="bg-transparent text-xs text-foreground outline-none cursor-pointer">
              <option value="zh">简中</option>
              <option value="en">EN</option>
              <option value="de">DE</option>
            </select>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground px-4 py-2 bg-secondary/50 rounded-full border border-border">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          {t('engine.online')}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto bg-background p-6">
        <Outlet />
      </main>
    </div>
  );
}
