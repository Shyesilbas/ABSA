import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
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

function linesToItems(text: string, max: number) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length >= 2);
  return lines.slice(0, max).map((line, i) => ({ text: line, id: i }));
}

function sentimentClass(s: string): string {
  if (s === "Negative" || s === "Neutral" || s === "Positive") return s;
  return "Neutral";
}

export default function App() {
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
  const singleChars = singleText.trim().length;
  const batchLineCount = linesToItems(batchInput, maxBatch).length;

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
      setErr("En az bir satır (en az 2 karakter) girin.");
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
      setErr("Lutfen bir CSV dosyasi secin.");
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
        setErr("Grafik için önce toplu metin alanına satır girin veya toplu tahmin çalıştırın.");
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
        setErr("Grafik için metin satırları gerekli.");
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

  const healthBadge = () => {
    if (!health) return <span className="badge">API bekleniyor…</span>;
    if (health.status === "ok" && health.model_loaded)
      return <span className="badge ok">Model hazır</span>;
    if (health.status === "degraded")
      return <span className="badge err">Model yok / hata</span>;
    return <span className="badge warn">Başlatılıyor…</span>;
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-main">
          <h1>Türkçe duygu analizi</h1>
          <p className="sub">Backend REST API ile bağlı, modernleştirilmiş demo arayüz</p>
          {meta && (
            <div className="meta-grid">
              <div>
                Model: <strong>{meta.model_name}</strong>
              </div>
              <div>
                Fallback:{" "}
                <strong>
                  {meta.confidence_fallback_enabled
                    ? `≤${meta.confidence_threshold} → ${meta.confidence_fallback_label}`
                    : "kapalı"}
                </strong>
              </div>
              <div>
                Toplu limit: <strong>{meta.max_batch_items}</strong> satır
              </div>
            </div>
          )}
        </div>
        <div className="badges">
          {healthBadge()}
          <button type="button" className="btn-secondary" onClick={() => void refreshStatus()}>
            Durumu yenile
          </button>
        </div>
      </header>

      <section className="quick-stats">
        <div className="stat-card">
          <span className="hint">Tekil karakter</span>
          <strong>{singleChars}</strong>
        </div>
        <div className="stat-card">
          <span className="hint">Toplu satır</span>
          <strong>{batchLineCount}</strong>
        </div>
        <div className="stat-card">
          <span className="hint">Maksimum toplu limit</span>
          <strong>{maxBatch}</strong>
        </div>
        <div className="stat-card">
          <span className="hint">Maksimum CSV boyutu</span>
          <strong>{uploadSizeLabel}</strong>
        </div>
      </section>

      {err && (
        <div className="error" role="alert">
          {err}
        </div>
      )}

      <div className="grid">
        <section className="card">
          <h2>Tekil tahmin</h2>
          <p className="hint">Bir cümle girin, model duygu ve güven skorlarını üretsin.</p>
          <textarea
            value={singleText}
            onChange={(e) => setSingleText(e.target.value)}
            placeholder="Örn: Kargo gününde geldi, çok memnun kaldım."
            maxLength={8000}
          />
          <div className="actions">
            <button type="button" className="btn-primary" disabled={singleLoading} onClick={() => void onSinglePredict()}>
              {singleLoading ? "…" : "Tahmin et"}
            </button>
          </div>
          {singleRes && (
            <div className="result">
              <div className={`sentiment-pill ${sentimentClass(singleRes.sentiment)}`}>{singleRes.sentiment}</div>
              <p className="hint">
                Ham: {singleRes.raw_sentiment} · Güven: {singleRes.confidence.toFixed(4)}
                {singleRes.fallback_applied ? " · Fallback uygulandı" : ""}
              </p>
              {Object.entries(singleRes.probabilities).map(([k, v]) => (
                <div key={k} className="prob-row">
                  <span>{k}</span>
                  <div className="prob-bar">
                    <i style={{ width: `${Math.min(100, v * 100)}%` }} />
                  </div>
                  <span>{(v * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <h2>Toplu tahmin</h2>
          <div className="upload-box">
            <label className="hint" htmlFor="batch-upload-input">
              CSV yukle (text kolonu gerekli)
            </label>
            <input
              id="batch-upload-input"
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => setBatchFile(e.target.files?.[0] ?? null)}
            />
            <p className="hint">Dosya limiti: {uploadSizeLabel}. Maksimum satir: {maxBatch}.</p>
            <div className="actions">
              <button type="button" className="btn-primary" disabled={uploadLoading} onClick={() => void onBatchUpload()}>
                {uploadLoading ? "…" : "CSV yukle ve tahmin et"}
              </button>
            </div>
          </div>

          <div className="divider" />
          <textarea
            value={batchInput}
            onChange={(e) => setBatchInput(e.target.value)}
            placeholder={"Her satır bir metin (en az 2 karakter).\nMaksimum satır: API /meta."}
          />
          <p className="hint">Boş satırlar atlanır. Şu an en fazla {maxBatch} satır gönderilir.</p>
          <div className="actions">
            <button type="button" className="btn-primary" disabled={batchLoading} onClick={() => void onBatchPredict()}>
              {batchLoading ? "…" : "Toplu tahmin"}
            </button>
          </div>
          {batchRows && batchRows.length > 0 && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Metin</th>
                    <th>Duygu</th>
                    <th>Güven</th>
                  </tr>
                </thead>
                <tbody>
                  {batchRows.map((row, i) => (
                    <tr key={i}>
                      <td>{row.id ?? i}</td>
                      <td>{row.text.length > 80 ? `${row.text.slice(0, 80)}…` : row.text}</td>
                      <td>
                        <span className={`sentiment-chip ${sentimentClass(row.sentiment)}`}>{row.sentiment}</span>
                      </td>
                      <td>{row.confidence.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {batchRows && batchRows.length === 0 && <p className="hint">Toplu sonuç bulunamadı.</p>}
        </section>

        <section className="card full-width">
          <h2>Dağılım grafiği</h2>
          <p className="hint">
            Son toplu tahmin sonuçlarına göre (ek model çağrısı yok) veya aynı metin kutusundaki satırlara göre (sunucuda yeniden tahmin) grafik üretir.
          </p>
          <div className="viz-input-grid">
            <input value={vizTitle} onChange={(e) => setVizTitle(e.target.value)} placeholder="Grafik başlığı (isteğe bağlı)" />
            <input value={vizKw} onChange={(e) => setVizKw(e.target.value)} placeholder="Alt başlık / anahtar kelimeler (isteğe bağlı)" />
          </div>
          <div className="actions">
            <button
              type="button"
              className="btn-secondary"
              disabled={statsLoading || !rowsFromBatch?.length}
              onClick={() => void fetchStats("rows")}
            >
              {statsLoading ? "…" : "İstatistik (tahmin sonucu)"}
            </button>
            <button
              type="button"
              className="btn-secondary"
              disabled={pngLoading || !rowsFromBatch?.length}
              onClick={() => void fetchPng("rows")}
            >
              {pngLoading ? "…" : "PNG (tahmin sonucu)"}
            </button>
            <button type="button" className="btn-secondary" disabled={statsLoading} onClick={() => void fetchStats("texts")}>
              İstatistik (metin kutusu)
            </button>
            <button type="button" className="btn-secondary" disabled={pngLoading} onClick={() => void fetchPng("texts")}>
              PNG (metin kutusu)
            </button>
          </div>
          {stats && (
            <div className="stats-list">
              <span>
                Toplam: <strong>{stats.total}</strong>
              </span>
              {Object.entries(stats.counts).map(([k, v]) => (
                <span key={k}>
                  {k}: <strong>{v}</strong>
                </span>
              ))}
              <span className="hint">({stats.source})</span>
            </div>
          )}
          {chartUrl && <img className="chart-preview" src={chartUrl} alt="Duygu dağılımı" />}
        </section>
      </div>
    </div>
  );
}
