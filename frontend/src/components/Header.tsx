import { HealthResponse } from '../api';
import { RefreshCcw, Activity, ShieldCheck, ShieldAlert, Zap, Moon, Sun } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface HeaderProps {
  health: HealthResponse | null;
  theme: "dark" | "light";
  setTheme: (t: "dark" | "light") => void;
  onRefresh: () => void;
}

export default function Header({ health, theme, setTheme, onRefresh }: HeaderProps) {
  const renderBadge = () => {
    if (!health) {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-100 dark:bg-slate-800 px-3 py-1 text-xs font-medium text-slate-500 dark:text-slate-300">
          <Activity className="h-3.5 w-3.5 animate-pulse" />
          Connecting...
        </span>
      );
    }
    if (health.status === "ok" && health.model_loaded) {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-700">
          <ShieldCheck className="h-3.5 w-3.5" />
          System Ready
        </span>
      );
    }
    if (health.status === "degraded") {
      return (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-100 px-3 py-1 text-xs font-medium text-rose-700">
          <ShieldAlert className="h-3.5 w-3.5" />
          Model Error
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700">
        <Zap className="h-3.5 w-3.5" />
        Starting...
      </span>
    );
  };

  return (
    <div role="banner" className="mb-8 flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">Turkish Sentiment Analysis</h1>
        <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
          Enterprise-grade analysis dashboard integrated with REST API.
        </p>
      </div>

      <div className="flex items-center gap-3">
        {renderBadge()}
        
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="inline-flex items-center justify-center rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-2 text-slate-500 dark:text-slate-300 shadow-sm transition-colors hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-slate-500 dark:focus:ring-slate-700"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title="Toggle Theme"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>

        <button
          onClick={onRefresh}
          aria-label="Refresh system health status"
          className="inline-flex items-center gap-2 rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm font-medium text-slate-500 dark:text-slate-300 shadow-sm transition-colors hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-slate-500 dark:focus:ring-slate-700"
        >
          <RefreshCcw className="h-4 w-4" />
          Refresh
        </button>
      </div>
    </div>
  );
}
