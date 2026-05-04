import { useCallback, useEffect, useMemo, useState } from "react";
import {
  apiJson,
  apiPostBlob,
  apiUploadBatchCsv,
  type BatchPredictionRow,
  type DistributionStatsResponse,
  type HealthResponse,
  type MetaResponse,
  type PredictResponse,
} from "./api";
import Header, { cn } from "./components/Header";
import SinglePredictionCard from "./components/SinglePredictionCard";
import BatchPredictionCard from "./components/BatchPredictionCard";
import { AlertCircle } from "lucide-react";

function linesToItems(text: string, max: number) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length >= 2);
  return lines.slice(0, max).map((line, i) => ({ text: line, id: i }));
}

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [theme]);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [singleText, setSingleText] = useState("");
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleRes, setSingleRes] = useState<PredictResponse | null>(null);

  const [batchInput, setBatchInput] = useState("");
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [batchRows, setBatchRows] = useState<BatchPredictionRow[] | null>(null);

  const [vizTitle, setVizTitle] = useState("");
  const [vizKw, setVizKw] = useState("");
  const [statsLoading, setStatsLoading] = useState(false);
  const [pngLoading, setPngLoading] = useState(false);
  const [stats, setStats] = useState<DistributionStatsResponse | null>(null);
  const [chartUrl, setChartUrl] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    setErr(null);
    try {
      const [h, m] = await Promise.all([
        apiJson<HealthResponse>("/health"),
        apiJson<MetaResponse>("/meta"),
      ]);
      setHealth(h);
      setMeta(m);
    } catch (e) {
      setHealth(null);
      setMeta(null);
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const t = setInterval(() => void refreshStatus(), 20000);
    return () => clearInterval(t);
  }, [refreshStatus]);

  useEffect(() => {
    return () => {
      if (chartUrl) URL.revokeObjectURL(chartUrl);
    };
  }, [chartUrl]);

  const maxBatch = meta?.max_batch_items ?? 500;
  const maxUploadBytes = meta?.max_batch_upload_bytes ?? 1024 * 1024;

  const uploadSizeLabel = useMemo(() => {
    if (maxUploadBytes >= 1024 * 1024) return `${(maxUploadBytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${Math.round(maxUploadBytes / 1024)} KB`;
  }, [maxUploadBytes]);

  const onSinglePredict = async () => {
    const t = singleText.trim();
    if (!t) return;
    setSingleLoading(true);
    setErr(null);
    setSingleRes(null);
    try {
      const r = await apiJson<PredictResponse>("/predict", {
        method: "POST",
        body: JSON.stringify({ text: t }),
      });
      setSingleRes(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSingleLoading(false);
    }
  };

  const onBatchPredict = async () => {
    const items = linesToItems(batchInput, maxBatch);
    if (!items.length) {
      setErr("Enter at least one line (min 2 characters).");
      return;
    }
    setBatchLoading(true);
    setErr(null);
    setBatchRows(null);
    setStats(null);
    if (chartUrl) URL.revokeObjectURL(chartUrl);
    setChartUrl(null);
    try {
      const r = await apiJson<{ predictions: BatchPredictionRow[] }>("/predict/batch", {
        method: "POST",
        body: JSON.stringify({ items }),
      });
      setBatchRows(r.predictions);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBatchLoading(false);
    }
  };

  const onBatchUpload = async () => {
    if (!batchFile) {
      setErr("Please select a CSV file.");
      return;
    }
    setUploadLoading(true);
    setErr(null);
    setBatchRows(null);
    setStats(null);
    if (chartUrl) URL.revokeObjectURL(chartUrl);
    setChartUrl(null);
    try {
      const r = await apiUploadBatchCsv(batchFile, {
        topicTitle: vizTitle,
        keywordsSubtitle: vizKw,
      });
      setBatchRows(r.predictions);
      setStats({
        counts: r.counts,
        total: r.total,
        topic_title: r.topic_title,
        keywords_subtitle: r.keywords_subtitle,
        source: r.source,
      });

      const rows = r.predictions.map((x) => ({ sentiment: x.sentiment }));
      if (rows.length) {
        const blob = await apiPostBlob("/visualize/distribution", {
          topic_title: r.topic_title,
          keywords_subtitle: r.keywords_subtitle,
          rows,
        });
        setChartUrl(URL.createObjectURL(blob));
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setUploadLoading(false);
    }
  };

  const vizBodyBase = useMemo(() => {
    const o: Record<string, unknown> = {};
    if (vizTitle.trim()) o.topic_title = vizTitle.trim();
    if (vizKw.trim()) o.keywords_subtitle = vizKw.trim();
    return o;
  }, [vizTitle, vizKw]);

  const rowsFromBatch = useMemo(() => {
    if (!batchRows?.length) return null;
    return batchRows.map((r) => ({ sentiment: r.sentiment }));
  }, [batchRows]);

  const fetchStats = async (source: "rows" | "texts") => {
    setStatsLoading(true);
    setErr(null);
    try {
      const body =
        source === "rows" && rowsFromBatch
          ? { ...vizBodyBase, rows: rowsFromBatch }
          : { ...vizBodyBase, texts: linesToItems(batchInput, maxBatch).map((x) => x.text) };
      if (source === "texts" && !(body as { texts?: string[] }).texts?.length) {
        setErr("Enter lines in the batch text area or run a batch prediction first.");
        return;
      }
      const s = await apiJson<DistributionStatsResponse>("/visualize/distribution/stats", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setStats(s);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchPng = async (source: "rows" | "texts") => {
    setPngLoading(true);
    setErr(null);
    try {
      const body =
        source === "rows" && rowsFromBatch
          ? { ...vizBodyBase, rows: rowsFromBatch }
          : { ...vizBodyBase, texts: linesToItems(batchInput, maxBatch).map((x) => x.text) };
      if (source === "texts" && !(body as { texts?: string[] }).texts?.length) {
        setErr("Text lines are required for the chart.");
        return;
      }
      const blob = await apiPostBlob("/visualize/distribution", body);
      if (chartUrl) URL.revokeObjectURL(chartUrl);
      setChartUrl(URL.createObjectURL(blob));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPngLoading(false);
    }
  };

  const [activeTab, setActiveTab] = useState<"single" | "batch">("single");

  return (
    <div className="min-h-screen bg-white text-slate-900 transition-colors duration-300 dark:bg-slate-950 dark:text-slate-200 py-8 px-4 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">
        <Header health={health} theme={theme} setTheme={setTheme} onRefresh={refreshStatus} />
        
        {err && (
          <div className="mb-8 rounded-lg border border-rose-500/20 bg-rose-500/10 p-4 shadow-sm">
            <div className="flex">
              <div className="flex-shrink-0">
                <AlertCircle className="h-5 w-5 text-rose-500" aria-hidden="true" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-rose-600 dark:text-rose-400">Error</h3>
                <div className="mt-1 text-sm text-rose-600/80 dark:text-rose-400/80">{err}</div>
              </div>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="mb-8 border-b border-slate-200 dark:border-slate-800">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab("single")}
              className={cn(
                "whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors",
                activeTab === "single"
                  ? "border-slate-900 text-slate-900 dark:border-white dark:text-white"
                  : "border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 hover:border-slate-300 dark:hover:border-slate-700"
              )}
            >
              Single Prediction
            </button>
            <button
              onClick={() => setActiveTab("batch")}
              className={cn(
                "whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors",
                activeTab === "batch"
                  ? "border-slate-900 text-slate-900 dark:border-white dark:text-white"
                  : "border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 hover:border-slate-300 dark:hover:border-slate-700"
              )}
            >
              Batch Prediction & Analysis
            </button>
          </nav>
        </div>

        <div className="space-y-8">
          {activeTab === "single" ? (
            <SinglePredictionCard
              singleText={singleText}
              setSingleText={setSingleText}
              singleLoading={singleLoading}
              onSinglePredict={onSinglePredict}
              singleRes={singleRes}
            />
          ) : (
            <BatchPredictionCard
              batchFile={batchFile}
              setBatchFile={setBatchFile}
              uploadSizeLabel={uploadSizeLabel}
              maxBatch={maxBatch}
              uploadLoading={uploadLoading}
              onBatchUpload={onBatchUpload}
              batchInput={batchInput}
              setBatchInput={setBatchInput}
              batchLoading={batchLoading}
              onBatchPredict={onBatchPredict}
              batchRows={batchRows}
              vizTitle={vizTitle}
              setVizTitle={setVizTitle}
              vizKw={vizKw}
              setVizKw={setVizKw}
              statsLoading={statsLoading}
              pngLoading={pngLoading}
              stats={stats}
              chartUrl={chartUrl}
              fetchStats={fetchStats}
              fetchPng={fetchPng}
            />
          )}
        </div>
      </div>
    </div>
  );
}
