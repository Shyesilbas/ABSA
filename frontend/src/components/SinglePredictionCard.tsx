import { PredictResponse } from '../api';
import { MessageSquare, Loader2 } from 'lucide-react';
import { cn } from './Header';

interface SinglePredictionCardProps {
  singleText: string;
  setSingleText: (text: string) => void;
  singleLoading: boolean;
  onSinglePredict: () => void;
  singleRes: PredictResponse | null;
}

export function sentimentColor(s: string, type: 'bg' | 'text' | 'border' | 'fill' = 'bg'): string {
  const isPos = s === "Positive";
  const isNeg = s === "Negative";
  
  if (type === 'bg') return isPos ? 'bg-emerald-500/10 dark:bg-emerald-500/10' : isNeg ? 'bg-rose-500/10 dark:bg-rose-500/10' : 'bg-slate-100 dark:bg-slate-800';
  if (type === 'text') return isPos ? 'text-emerald-600 dark:text-emerald-400' : isNeg ? 'text-rose-600 dark:text-rose-400' : 'text-slate-500 dark:text-slate-400';
  if (type === 'border') return isPos ? 'border-emerald-500/20' : isNeg ? 'border-rose-500/20' : 'border-slate-200 dark:border-slate-700';
  if (type === 'fill') return isPos ? 'bg-emerald-500' : isNeg ? 'bg-rose-500' : 'bg-slate-500';
  
  return 'bg-slate-100 dark:bg-slate-800';
}

export default function SinglePredictionCard({
  singleText,
  setSingleText,
  singleLoading,
  onSinglePredict,
  singleRes,
}: SinglePredictionCardProps) {
  return (
    <div role="region" aria-label="Single Prediction" className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 shadow-sm transition-colors duration-300">
      <div className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/30 px-6 py-4">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-slate-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Single Prediction</h2>
        </div>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Enter a sentence to analyze sentiment and confidence scores.
        </p>
      </div>
      
      <div className="flex-1 p-6">
        <textarea
          value={singleText}
          onChange={(e) => setSingleText(e.target.value)}
          placeholder="e.g. The service was excellent and the food was delicious."
          maxLength={8000}
          aria-label="Enter a Turkish sentence for sentiment analysis"
          className="h-32 w-full resize-none rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-950 p-3 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-500 focus:border-slate-400 dark:focus:border-slate-700 focus:outline-none focus:ring-1 focus:ring-slate-400 dark:focus:ring-slate-700 transition-colors"
        />
        
        <div className="mt-4 flex justify-end">
          <button
            type="button"
            disabled={singleLoading || !singleText.trim()}
            onClick={onSinglePredict}
            className="flex items-center justify-center gap-2 rounded-lg bg-slate-900 dark:bg-slate-100 px-4 py-2 text-sm font-medium text-white dark:text-slate-900 transition-colors hover:bg-slate-800 dark:hover:bg-white focus:outline-none focus:ring-2 focus:ring-slate-500 dark:focus:ring-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {singleLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            {singleLoading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {singleRes && (
          <div role="status" aria-live="polite" className="mt-6 rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/50 p-4">
            <div className="flex items-start justify-between">
              <div>
                <span className={cn(
                  "inline-flex items-center rounded-full px-2.5 py-0.5 text-sm font-semibold",
                  sentimentColor(singleRes.sentiment, 'bg'),
                  sentimentColor(singleRes.sentiment, 'text')
                )}>
                  {singleRes.sentiment}
                </span>
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  Raw: {singleRes.raw_sentiment} • Confidence: {singleRes.confidence.toFixed(4)}
                  {singleRes.fallback_applied && " • Fallback applied"}
                </p>
              </div>
            </div>

            <div className="mt-6 flex flex-wrap gap-6 border-t border-slate-200 dark:border-slate-800 pt-4">
              {Object.entries(singleRes.probabilities).map(([k, v]) => (
                <div key={k} className="flex flex-col">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 font-bold">{k}</span>
                  <span className={cn("text-sm font-semibold", v > 0.1 ? sentimentColor(k, 'text') : "text-slate-400 dark:text-slate-600")}>
                    {(v * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
