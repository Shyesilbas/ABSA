import { useState } from 'react';
import { BatchPredictionRow, DistributionStatsResponse } from '../api';
import { UploadCloud, ListChecks, Loader2, PieChart, BarChart3, Image as ImageIcon, FileText } from 'lucide-react';
import { sentimentColor } from './SinglePredictionCard';
import { cn } from './Header';

interface BatchPredictionCardProps {
  batchFile: File | null;
  setBatchFile: (f: File | null) => void;
  uploadSizeLabel: string;
  maxBatch: number;
  uploadLoading: boolean;
  onBatchUpload: () => void;
  batchInput: string;
  setBatchInput: (s: string) => void;
  batchLoading: boolean;
  onBatchPredict: () => void;
  batchRows: BatchPredictionRow[] | null;
  // Visualization props
  vizTitle: string;
  setVizTitle: (t: string) => void;
  vizKw: string;
  setVizKw: (k: string) => void;
  statsLoading: boolean;
  pngLoading: boolean;
  stats: DistributionStatsResponse | null;
  chartUrl: string | null;
  fetchStats: (source: "rows" | "texts") => void;
  fetchPng: (source: "rows" | "texts") => void;
}

export default function BatchPredictionCard({
  batchFile,
  setBatchFile,
  uploadSizeLabel,
  maxBatch,
  uploadLoading,
  onBatchUpload,
  batchInput,
  setBatchInput,
  batchLoading,
  onBatchPredict,
  batchRows,
  vizTitle,
  setVizTitle,
  vizKw,
  setVizKw,
  statsLoading,
  pngLoading,
  stats,
  chartUrl,
  fetchStats,
  fetchPng,
}: BatchPredictionCardProps) {
  const [resultFilter, setResultFilter] = useState<string>("All");
  const hasRows = batchRows && batchRows.length > 0;

  const filteredRows = (batchRows || []).filter((r) => {
    if (resultFilter === "All") return true;
    return r.sentiment === resultFilter;
  });

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 shadow-sm transition-colors duration-300">
      <div className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/30 px-6 py-4">
        <div className="flex items-center gap-2">
          <ListChecks className="h-5 w-5 text-slate-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Batch Prediction</h2>
        </div>
      </div>

      <div className="p-6">
        {/* Input Methods */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* CSV Upload Section */}
          <div className="rounded-lg border border-dashed border-slate-300 dark:border-slate-700 p-6 text-center transition-colors hover:border-slate-400 dark:hover:border-slate-600 bg-slate-50/50 dark:bg-transparent">
            <UploadCloud className="mx-auto h-8 w-8 text-slate-400 dark:text-slate-500" />
            <div className="mt-4 flex text-sm leading-6 text-slate-600 dark:text-slate-400 justify-center">
              <label
                htmlFor="batch-upload-input"
                className="relative cursor-pointer rounded-md bg-transparent font-semibold text-slate-900 dark:text-white focus-within:outline-none focus-within:ring-2 focus-within:ring-slate-500 dark:focus-within:ring-slate-700 focus-within:ring-offset-2 focus-within:ring-offset-white dark:focus-within:ring-offset-slate-900 hover:text-slate-700 dark:hover:text-slate-200"
              >
                <span>Upload CSV</span>
                <input
                  id="batch-upload-input"
                  name="batch-upload-input"
                  type="file"
                  accept=".csv,text/csv"
                  className="sr-only"
                  onChange={(e) => setBatchFile(e.target.files?.[0] ?? null)}
                />
              </label>
              <p className="pl-1 text-slate-400 dark:text-slate-500">or drag and drop</p>
            </div>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Limit: {uploadSizeLabel}.</p>
            {batchFile && (
              <p className="mt-2 text-sm font-medium text-emerald-600 dark:text-emerald-400 truncate">
                {batchFile.name}
              </p>
            )}
            <div className="mt-4">
              <button
                type="button"
                disabled={uploadLoading || !batchFile}
                onClick={onBatchUpload}
                className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-slate-900 dark:bg-slate-100 px-4 py-2 text-sm font-medium text-white dark:text-slate-900 transition-colors hover:bg-slate-800 dark:hover:bg-white focus:outline-none focus:ring-2 focus:ring-slate-500 dark:focus:ring-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploadLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
                {uploadLoading ? "Processing..." : "Upload and Predict"}
              </button>
            </div>
          </div>

          {/* Text Input Section */}
          <div className="flex flex-col">
            <textarea
              value={batchInput}
              onChange={(e) => setBatchInput(e.target.value)}
              placeholder={`One text per line (max ${maxBatch} lines).`}
              className="flex-1 min-h-[120px] w-full resize-none rounded-lg border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-950 p-3 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-500 focus:border-slate-400 dark:focus:border-slate-700 focus:outline-none focus:ring-1 focus:ring-slate-400 dark:focus:ring-slate-700 transition-colors"
            />
            <div className="mt-3 flex justify-end">
              <button
                type="button"
                disabled={batchLoading || !batchInput.trim()}
                onClick={onBatchPredict}
                className="flex items-center justify-center gap-2 rounded-lg bg-slate-900 dark:bg-slate-100 px-4 py-2 text-sm font-medium text-white dark:text-slate-900 transition-colors hover:bg-slate-800 dark:hover:bg-white focus:outline-none focus:ring-2 focus:ring-slate-500 dark:focus:ring-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {batchLoading && <Loader2 className="h-4 w-4 animate-spin" />}
                {batchLoading ? "Analyzing..." : "Batch Predict"}
              </button>
            </div>
          </div>
        </div>

        {/* Visualization & Stats Section (Only if results exist) */}
        {hasRows && (
          <div className="mt-8 rounded-xl border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/20 p-6">
            <div className="flex items-center gap-2 mb-6">
              <PieChart className="h-5 w-5 text-slate-400" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Visualization & Analysis</h3>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 mb-6">
              <input
                type="text"
                value={vizTitle}
                onChange={(e) => setVizTitle(e.target.value)}
                placeholder="Chart Title (Optional)"
                className="block w-full rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-950 px-3 py-2 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-500 focus:border-slate-400 dark:focus:border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400 dark:focus:ring-slate-600 transition-colors"
              />
              <input
                type="text"
                value={vizKw}
                onChange={(e) => setVizKw(e.target.value)}
                placeholder="Subtitle / Keywords (Optional)"
                className="block w-full rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-950 px-3 py-2 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-500 focus:border-slate-400 dark:focus:border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400 dark:focus:ring-slate-600 transition-colors"
              />
            </div>

            <div className="flex flex-wrap gap-3 mb-6">
              <button
                type="button"
                disabled={statsLoading}
                onClick={() => fetchStats("rows")}
                className="inline-flex items-center gap-2 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 px-4 py-2 text-sm font-medium text-slate-600 dark:text-slate-300 shadow-sm transition-colors hover:bg-slate-50 dark:hover:bg-slate-700 hover:text-slate-900 dark:hover:text-white"
              >
                {statsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <BarChart3 className="h-4 w-4" />}
                Show Statistics
              </button>
              <button
                type="button"
                disabled={pngLoading}
                onClick={() => fetchPng("rows")}
                className="inline-flex items-center gap-2 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 px-4 py-2 text-sm font-medium text-slate-600 dark:text-slate-300 shadow-sm transition-colors hover:bg-slate-50 dark:hover:bg-slate-700 hover:text-slate-900 dark:hover:text-white"
              >
                {pngLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ImageIcon className="h-4 w-4" />}
                Generate Chart (PNG)
              </button>
            </div>

            {/* Results Display Area */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {stats && (
                <div className="rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
                  <h4 className="text-sm font-medium text-slate-900 dark:text-white mb-3 flex items-center gap-2">
                    <FileText className="h-4 w-4 text-slate-400" />
                    Distribution Data
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm py-1 border-b border-slate-100 dark:border-slate-800/50">
                      <span className="text-slate-500 dark:text-slate-400">Total</span>
                      <span className="text-slate-900 dark:text-white font-bold">{stats.total}</span>
                    </div>
                    {Object.entries(stats.counts).map(([k, v]) => (
                      <div key={k} className="flex justify-between text-sm py-1 border-b border-slate-100 dark:border-slate-800/50 last:border-0">
                        <span className="text-slate-500 dark:text-slate-400">{k}</span>
                        <span className="text-slate-900 dark:text-white font-bold">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {chartUrl && (
                <div className="rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2 flex items-center justify-center shadow-sm">
                  <img src={chartUrl} alt="Distribution" className="max-w-full rounded-md opacity-90 dark:opacity-80" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results Table Section */}
        {hasRows && (
          <div className="mt-8">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-4">
              <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400">Prediction Results</h3>
              
              {/* Sentiment Filter Tabs */}
              <div className="flex p-1 bg-slate-100 dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 transition-colors">
                {["All", "Positive", "Negative", "Neutral"].map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setResultFilter(filter)}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium rounded-md transition-all",
                      resultFilter === filter
                        ? "bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm"
                        : "text-slate-500 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                    )}
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-transparent">
              <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800">
                <thead className="bg-slate-50 dark:bg-slate-800/50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase text-slate-400">#</th>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase text-slate-400">Text</th>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase text-slate-400">Sentiment</th>
                    <th className="px-6 py-3 text-right text-xs font-medium uppercase text-slate-400">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                  {filteredRows.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                      <td className="px-6 py-4 text-sm text-slate-400 dark:text-slate-500">{row.id ?? i}</td>
                      <td className="px-6 py-4 text-sm text-slate-600 dark:text-slate-300">
                        {row.text.length > 100 ? `${row.text.slice(0, 100)}…` : row.text}
                      </td>
                      <td className="px-6 py-4">
                        <span className={cn(
                          "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
                          sentimentColor(row.sentiment, 'bg'),
                          sentimentColor(row.sentiment, 'text')
                        )}>
                          {row.sentiment}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right text-sm text-slate-400 dark:text-slate-500">
                        {row.confidence.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                  {filteredRows.length === 0 && (
                    <tr>
                      <td colSpan={4} className="px-6 py-12 text-center text-sm text-slate-400 dark:text-slate-500 italic">
                        No results found in this category.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>

  );
}

