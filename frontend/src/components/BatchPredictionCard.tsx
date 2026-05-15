import { useState } from 'react';
import { BatchPredictionRow, DistributionStatsResponse } from '../api';
import { UploadCloud, ListChecks, Loader2, PieChart, BarChart3, Image as ImageIcon, FileText, Download, DownloadCloud, Search, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import { PieChart as RePie, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
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
  progress: number;
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
  progress,
  fetchStats,
  fetchPng,
}: BatchPredictionCardProps) {
  const [resultFilter, setResultFilter] = useState<string>("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);

  const hasRows = batchRows && batchRows.length > 0;

  const downloadStatsCsv = () => {
    if (!stats) return;
    const lines = [];
    if (vizTitle) lines.push(`Title,${vizTitle}`);
    if (vizKw) lines.push(`Keywords,${vizKw}`);
    if (vizTitle || vizKw) lines.push("");

    lines.push("Label,Count");
    lines.push(`Total,${stats.total}`);
    Object.entries(stats.counts).forEach(([k, v]) => lines.push(`${k},${v}`));

    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `statistics_${new Date().getTime()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadResultsCsv = () => {
    if (!batchRows) return;
    const lines = [];
    if (vizTitle) lines.push(`Title,${vizTitle}`);
    if (vizKw) lines.push(`Keywords,${vizKw}`);
    if (vizTitle || vizKw) lines.push("");

    const headers = ["ID", "Text", "Sentiment", "Confidence"];
    lines.push(headers.join(","));
    
    batchRows.forEach(r => {
      lines.push([
        r.id ?? "",
        `"${r.text.replace(/"/g, '""')}"`,
        r.sentiment,
        r.confidence
      ].join(","));
    });

    const csvContent = lines.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `results_${new Date().getTime()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadChartPng = () => {
    if (!chartUrl) return;
    const a = document.createElement("a");
    a.href = chartUrl;
    a.download = `distribution_${new Date().getTime()}.png`;
    a.click();
  };

  const filteredRows = (batchRows || [])
    .filter((r) => {
      const matchesSentiment = resultFilter === "All" || r.sentiment === resultFilter;
      const matchesSearch = r.text.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesSentiment && matchesSearch;
    })
    .sort((a, b) => {
      if (!sortConfig) return 0;
      const { key, direction } = sortConfig;
      const aVal = a[key as keyof BatchPredictionRow] ?? "";
      const bVal = b[key as keyof BatchPredictionRow] ?? "";
      
      if (aVal < bVal) return direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return direction === 'asc' ? 1 : -1;
      return 0;
    });

  const chartData = stats ? Object.entries(stats.counts).map(([name, value]) => ({ name, value })) : [];
  const SENTIMENT_COLORS: Record<string, string> = {
    "Positive": "#10b981", // emerald-500
    "Neutral": "#64748b",  // slate-500
    "Negative": "#ef4444"  // rose-500
  };

  const handleSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const SortIcon = ({ column }: { column: string }) => {
    if (sortConfig?.key !== column) return <ArrowUpDown className="ml-2 h-3 w-3" />;
    return sortConfig.direction === 'asc' ? <ArrowUp className="ml-2 h-3 w-3 text-slate-900 dark:text-white" /> : <ArrowDown className="ml-2 h-3 w-3 text-slate-900 dark:text-white" />;
  };

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 shadow-sm transition-colors duration-300">
      <div className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/30 px-6 py-4">
        <div className="flex items-center gap-2">
          <ListChecks className="h-5 w-5 text-slate-400" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Batch Prediction</h2>
        </div>
      </div>

      <div className="p-6">
        {/* Metadata / Title Section (Moved to top so user can set before predict) */}
        <div className="mb-8 p-4 rounded-lg bg-slate-50 dark:bg-slate-800/20 border border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2 mb-4">
            <PieChart className="h-4 w-4 text-slate-400" />
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider">Report Details</h3>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5 ml-1">Chart Title</label>
              <input
                type="text"
                value={vizTitle}
                onChange={(e) => setVizTitle(e.target.value)}
                placeholder="e.g., Q2 Customer Feedback"
                className="block w-full rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-950 px-3 py-2 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-500 focus:border-slate-400 dark:focus:border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400 dark:focus:ring-slate-600 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5 ml-1">Subtitle / Keywords</label>
              <input
                type="text"
                value={vizKw}
                onChange={(e) => setVizKw(e.target.value)}
                placeholder="e.g., Coffee, Price, Quality"
                className="block w-full rounded-md border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-950 px-3 py-2 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-500 focus:border-slate-400 dark:focus:border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-400 dark:focus:ring-slate-600 transition-colors"
              />
            </div>
          </div>
        </div>

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

            {/* Progress Bar (Visible during loading) */}
            {(batchLoading || uploadLoading) && (
              <div className="mb-6">
                <div className="flex justify-between mb-1">
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-400">Processing...</span>
                  <span className="text-xs font-medium text-slate-600 dark:text-slate-400">{progress}%</span>
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5">
                  <div 
                    className="bg-slate-900 dark:bg-white h-1.5 rounded-full transition-all duration-300" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}

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
                <div className="relative rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 shadow-sm flex flex-col">
                  <h4 className="text-sm font-medium text-slate-900 dark:text-white mb-3 flex items-center gap-2">
                    <FileText className="h-4 w-4 text-slate-400" />
                    {vizTitle || "Distribution Data"}
                  </h4>
                  {vizKw && <p className="text-xs text-slate-500 dark:text-slate-400 -mt-2 mb-3 px-6">{vizKw}</p>}
                  
                  <div className="flex-1 flex flex-col justify-between">
                    <div className="space-y-2 mb-4">
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
                    
                    {/* Interactive Chart using Recharts */}
                    <div className="h-[200px] w-full mt-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <RePie>
                          <Pie
                            data={chartData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            paddingAngle={5}
                            dataKey="value"
                            stroke="none"
                          >
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={SENTIMENT_COLORS[entry.name] || '#cbd5e1'} />
                            ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
                          />
                          <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px' }} />
                        </RePie>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="absolute top-4 right-4">
                    <button
                      onClick={downloadStatsCsv}
                      title="Download Statistics CSV"
                      className="p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                    >
                      <Download className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
              {chartUrl && (
                <div className="relative group rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2 flex flex-col items-center justify-center shadow-sm min-h-[300px]">
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">Generated PNG Asset</h4>
                  <img src={chartUrl} alt="Distribution" className="max-w-full max-h-[350px] rounded-md opacity-90 dark:opacity-80 shadow-md" />
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={downloadChartPng}
                      title="Download Chart PNG"
                      className="bg-white/90 dark:bg-slate-800/90 p-2 rounded-full shadow-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors"
                    >
                      <Download className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results Table Section */}
        {hasRows && (
          <div className="mt-8">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-4">
              <div className="flex flex-col">
                <div className="flex items-center gap-4">
                  <h3 className="text-sm font-medium text-slate-900 dark:text-white font-bold">{vizTitle || "Prediction Results"}</h3>
                  <button
                    onClick={downloadResultsCsv}
                    title="Download All Results CSV"
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 transition-colors"
                  >
                    <DownloadCloud className="h-3.5 w-3.5" />
                    Download Full CSV
                  </button>
                </div>
                {vizKw && <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{vizKw}</p>}
              </div>
              
              <div className="flex flex-wrap items-center gap-3">
                {/* Search Bar */}
                <div className="relative">
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search results..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 pr-4 py-2 w-full sm:w-64 text-sm rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 focus:border-slate-400 focus:outline-none transition-colors"
                  />
                </div>

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
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-transparent">
              <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800">
                <thead className="bg-slate-50 dark:bg-slate-800/50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase text-slate-400">#</th>
                    <th className="px-6 py-3 text-left text-xs font-medium uppercase text-slate-400">Text</th>
                    <th 
                      onClick={() => handleSort('sentiment')}
                      className="px-6 py-3 text-left text-xs font-medium uppercase text-slate-400 cursor-pointer hover:text-slate-600 dark:hover:text-slate-200"
                    >
                      <div className="flex items-center">
                        Sentiment
                        <SortIcon column="sentiment" />
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('confidence')}
                      className="px-6 py-3 text-right text-xs font-medium uppercase text-slate-400 cursor-pointer hover:text-slate-600 dark:hover:text-slate-200"
                    >
                      <div className="flex items-center justify-end">
                        Confidence
                        <SortIcon column="confidence" />
                      </div>
                    </th>
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

