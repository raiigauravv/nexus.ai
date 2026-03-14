"use client";

import { useState, useCallback, useRef } from "react";
import {
  Upload,
  RefreshCw,
  CheckCircle,
  Layers,
  Search,
  Star,
  Cpu,
  Tag,
  Palette,
  BarChart2,
  Info,
} from "lucide-react";

const BACKEND = "http://localhost:8000/api/v1";

interface ColorSwatch {
  hex: string;
  rgb: number[];
  percentage: number;
  hue: number;
  saturation: number;
  brightness: number;
}

interface Classification {
  label: string;
  confidence: number;
}

interface Statistics {
  brightness: number;
  brightness_pct: number;
  contrast: number;
  saturation: number;
  saturation_pct: number;
  sharpness: number;
}

interface Metadata {
  width: number;
  height: number;
  aspect_ratio: number;
  mode: string;
  format: string;
  megapixels: number;
  file_size_kb: number;
  exif: Record<string, string>;
}

interface VisionResult {
  filename: string;
  metadata: Metadata;
  statistics: Statistics;
  palette: ColorSwatch[];
  classification: Classification[];
  top_label: string;
  tags: string[];
  visuals: {
    thumbnail: string;
    edges: string;
  };
}

interface SearchProduct {
  id: string;
  name: string;
  category: string;
  price: number;
  rating: number;
  tags: string[];
  rank: number;
  visual_similarity: number;
  similarity_pct: number;
}

interface SearchResult {
  query_image: string;
  model: string;
  results: SearchProduct[];
  total_indexed: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  Electronics: "bg-blue-100 text-blue-700",
  Books: "bg-amber-100 text-amber-700",
  Clothing: "bg-purple-100 text-purple-700",
  "Home & Kitchen": "bg-emerald-100 text-emerald-700",
  Sports: "bg-orange-100 text-orange-700",
  Gaming: "bg-red-100 text-red-700",
  Beauty: "bg-pink-100 text-pink-700",
  Automotive: "bg-slate-100 text-slate-700",
};

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatBar({ label, value, color, suffix = "%" }: { label: string; value: number; color: string; suffix?: string; }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-600 font-medium">{label}</span>
        <span className="font-mono text-gray-700">{pct}{suffix}</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-700 ${color}`} style={{ width: `${Math.min(100, pct)}%` }} />
      </div>
    </div>
  );
}

function ConfBar({ label, confidence }: { label: string; confidence: number }) {
  const pct = Math.round(confidence * 100);
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-600 flex-1 truncate" title={label}>{label}</span>
      <div className="w-28 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full bg-indigo-500 rounded-full transition-all duration-700" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right text-xs font-mono font-semibold text-indigo-700">{pct}%</span>
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-gray-400">{label}</p>
      <p className="font-semibold text-gray-700 truncate" title={value}>{value}</p>
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────────

export default function VisionDashboard() {
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<VisionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeVisual, setActiveVisual] = useState<"original" | "edges">("original");
  const [activeMode, setActiveMode] = useState<"search" | "analyze">("search");
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // ── Image Analysis ─────────────────────────────────────────────────────────
  const processFile = useCallback(async (file: File) => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${BACKEND}/vision/analyze`, { method: "POST", body: form });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Analysis failed");
      }
      setResult(await resp.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  // ── CLIP Visual Search ─────────────────────────────────────────────────────
  const handleSearchFile = useCallback(async (file: File) => {
    setError(null);
    setSearchLoading(true);
    setSearchResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${BACKEND}/vision/search?top_n=5`, { method: "POST", body: form });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Search failed");
      }
      setSearchResult(await resp.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSearchLoading(false);
    }
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-6">

      {/* Mode Tabs */}
      <div className="flex bg-gray-100 rounded-xl p-1 gap-1 w-fit">
        <button
          onClick={() => setActiveMode("search")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeMode === "search" ? "bg-white text-indigo-700 shadow-sm" : "text-gray-500 hover:text-gray-700"
          }`}
        >
          <Search className="w-4 h-4" />
          Visual Product Search
          <span className="text-xs bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded font-semibold">CLIP</span>
        </button>
        <button
          onClick={() => setActiveMode("analyze")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeMode === "analyze" ? "bg-white text-teal-700 shadow-sm" : "text-gray-500 hover:text-gray-700"
          }`}
        >
          <Layers className="w-4 h-4" />
          Image Analysis
        </button>
      </div>

      {/* ══ CLIP VISUAL SEARCH ══════════════════════════════════════════════ */}
      {activeMode === "search" && (
        <div className="space-y-4">

          {/* Info banner */}
          <div className="bg-gradient-to-r from-violet-50 to-indigo-50 border border-indigo-100 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="w-4 h-4 text-indigo-500" />
              <p className="text-sm font-semibold text-indigo-800">CLIP ViT-B/32 Visual Similarity Search</p>
            </div>
            <p className="text-xs text-indigo-600">
              Upload any product image — CLIP encodes it into a 512-dim embedding and finds the most visually similar 
              items from the 25-product catalog via cosine similarity.
            </p>
          </div>

          {/* Upload zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleSearchFile(f); }}
            onClick={() => searchInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 ${
              isDragging ? "border-violet-500 bg-violet-50 scale-[1.01]"
              : searchResult ? "border-emerald-300 bg-emerald-50/40 hover:border-emerald-400"
              : "border-gray-200 bg-gray-50 hover:border-violet-300 hover:bg-violet-50/30"
            }`}
          >
            <input
              ref={searchInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleSearchFile(f); }}
            />
            {searchLoading ? (
              <div className="flex flex-col items-center gap-3">
                <RefreshCw className="w-10 h-10 text-violet-400 animate-spin" />
                <p className="text-sm font-medium text-violet-600">Running CLIP encoding...</p>
                <p className="text-xs text-gray-400">Embedding → 512-dim space → cosine similarity search</p>
              </div>
            ) : searchResult ? (
              <div className="flex flex-col items-center gap-2">
                <CheckCircle className="w-8 h-8 text-emerald-500" />
                <p className="text-sm font-semibold text-emerald-700">Found {searchResult.results.length} similar products</p>
                <p className="text-xs text-gray-400">Click or drag to search with another image</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-full bg-violet-100 flex items-center justify-center">
                  <Search className="w-7 h-7 text-violet-500" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-700">Drop a product image to search</p>
                  <p className="text-xs text-gray-400 mt-1">CLIP finds visually similar items · JPEG, PNG, WebP · max 15MB</p>
                </div>
              </div>
            )}
            {error && <p className="text-xs text-red-600 mt-3 font-medium">⚠ {error}</p>}
          </div>

          {/* Search Results */}
          {searchResult && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <Cpu className="w-3.5 h-3.5" />
                <span>Model: {searchResult.model} · {searchResult.total_indexed} products indexed in 512-dim embedding space</span>
              </div>
              {searchResult.results.map((prod) => (
                <div
                  key={prod.id}
                  className="bg-white border border-gray-100 rounded-xl p-4 flex items-center gap-4 hover:shadow-md hover:border-violet-100 transition-all"
                >
                  <div className="w-10 h-10 rounded-full bg-violet-100 flex items-center justify-center font-bold text-violet-600 text-sm flex-shrink-0">
                    #{prod.rank}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-gray-800 text-sm truncate">{prod.name}</p>
                    <div className="flex items-center gap-2 mt-1 flex-wrap">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${CATEGORY_COLORS[prod.category] || "bg-gray-100 text-gray-600"}`}>
                        {prod.category}
                      </span>
                      <span className="text-xs text-gray-500">${prod.price.toFixed(2)}</span>
                      <div className="flex items-center gap-0.5">
                        <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                        <span className="text-xs text-gray-600">{prod.rating}</span>
                      </div>
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className={`text-lg font-bold ${
                      prod.similarity_pct > 75 ? "text-emerald-600"
                      : prod.similarity_pct > 50 ? "text-indigo-600"
                      : "text-gray-500"
                    }`}>
                      {prod.similarity_pct.toFixed(1)}%
                    </div>
                    <p className="text-xs text-gray-400">similarity</p>
                    <div className="w-24 h-1.5 bg-gray-100 rounded-full mt-1 overflow-hidden">
                      <div className="h-full bg-violet-500 rounded-full" style={{ width: `${prod.similarity_pct}%` }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ══ IMAGE ANALYSIS ══════════════════════════════════════════════════ */}
      {activeMode === "analyze" && (
        <div className="space-y-6">

          {/* Upload Zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 ${
              isDragging ? "border-indigo-500 bg-indigo-50 scale-[1.01]"
              : result ? "border-emerald-300 bg-emerald-50/40 hover:border-emerald-400"
              : "border-gray-200 bg-gray-50 hover:border-indigo-300 hover:bg-indigo-50/30"
            }`}
          >
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            {loading ? (
              <div className="flex flex-col items-center gap-3">
                <RefreshCw className="w-10 h-10 text-indigo-400 animate-spin" />
                <p className="text-sm font-medium text-indigo-600">Running CV pipeline...</p>
                <p className="text-xs text-gray-400">Classification · Color extraction · Edge detection</p>
              </div>
            ) : result ? (
              <div className="flex flex-col items-center gap-2">
                <CheckCircle className="w-8 h-8 text-emerald-500" />
                <p className="text-sm font-semibold text-emerald-700">
                  Analysis complete — <span className="font-bold">{result.top_label}</span>
                </p>
                <p className="text-xs text-gray-400">Click or drag to analyze another image</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-full bg-indigo-100 flex items-center justify-center">
                  <Upload className="w-7 h-7 text-indigo-500" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-700">Drop an image here</p>
                  <p className="text-xs text-gray-400 mt-1">or click to browse · JPEG, PNG, WebP · max 15MB</p>
                </div>
              </div>
            )}
            {error && <p className="text-xs text-red-600 mt-3 font-medium">⚠ {error}</p>}
          </div>

          {/* Results */}
          {result && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

              {/* Left: Visual + Metadata */}
              <div className="lg:col-span-1 space-y-4">
                <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                  <div className="flex border-b border-gray-100">
                    {(["original", "edges"] as const).map((v) => (
                      <button
                        key={v}
                        onClick={() => setActiveVisual(v)}
                        className={`flex-1 text-xs py-2 font-medium transition-colors capitalize ${
                          activeVisual === v ? "text-indigo-700 bg-indigo-50 border-b-2 border-indigo-500" : "text-gray-400 hover:text-gray-600"
                        }`}
                      >
                        {v === "edges" ? "Edge Detection" : "Original"}
                      </button>
                    ))}
                  </div>
                  <div className="p-2 bg-gray-900 flex items-center justify-center min-h-48">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/${activeVisual === "edges" ? "png" : "jpeg"};base64,${
                        activeVisual === "edges" ? result.visuals.edges : result.visuals.thumbnail
                      }`}
                      alt={activeVisual === "edges" ? "Edge detection" : "Analyzed image"}
                      className="max-w-full max-h-64 object-contain rounded"
                    />
                  </div>
                </div>

                <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1">
                    <Info className="w-3.5 h-3.5" /> Image Metadata
                  </p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                    <MetaRow label="Dimensions" value={`${result.metadata.width} × ${result.metadata.height}`} />
                    <MetaRow label="Format" value={result.metadata.format || result.metadata.mode} />
                    <MetaRow label="Megapixels" value={`${result.metadata.megapixels} MP`} />
                    <MetaRow label="File Size" value={`${result.metadata.file_size_kb} KB`} />
                    <MetaRow label="Aspect" value={result.metadata.aspect_ratio.toFixed(2)} />
                    <MetaRow label="Color Mode" value={result.metadata.mode} />
                    {Object.entries(result.metadata.exif || {}).slice(0, 2).map(([k, v]) => (
                      <MetaRow key={k} label={k} value={String(v).slice(0, 20)} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Right: Analysis panels */}
              <div className="lg:col-span-2 space-y-4">

                {/* Classification */}
                <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <Layers className="w-4 h-4 text-indigo-500" />
                    <h3 className="text-sm font-semibold text-gray-800">Scene Classification</h3>
                    <span className="ml-auto text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-semibold border border-indigo-200">
                      {result.top_label}
                    </span>
                  </div>
                  <div className="space-y-2.5">
                    {result.classification.map((cls) => (
                      <ConfBar key={cls.label} label={cls.label} confidence={cls.confidence} />
                    ))}
                  </div>
                </div>

                {/* Stats */}
                <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <BarChart2 className="w-4 h-4 text-teal-500" />
                    <h3 className="text-sm font-semibold text-gray-800">Image Statistics</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                    <StatBar label="Brightness" value={result.statistics.brightness} color="bg-yellow-400" />
                    <StatBar label="Contrast" value={result.statistics.contrast} color="bg-blue-500" />
                    <StatBar label="Saturation" value={result.statistics.saturation} color="bg-pink-500" />
                    <StatBar label="Sharpness" value={result.statistics.sharpness} color="bg-emerald-500" />
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Color Palette */}
                  <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Palette className="w-4 h-4 text-purple-500" />
                      <h3 className="text-sm font-semibold text-gray-800">Dominant Colors</h3>
                    </div>
                    <div className="flex gap-1 mb-3 rounded-xl overflow-hidden h-10">
                      {result.palette.map((c) => (
                        <div
                          key={c.hex}
                          title={`${c.hex} — ${c.percentage}%`}
                          className="transition-all duration-300 hover:scale-y-110"
                          style={{ backgroundColor: c.hex, width: `${c.percentage}%` }}
                        />
                      ))}
                    </div>
                    <div className="space-y-1.5">
                      {result.palette.map((c) => (
                        <div key={c.hex} className="flex items-center gap-2">
                          <div className="w-5 h-5 rounded-md border border-gray-200 flex-shrink-0" style={{ backgroundColor: c.hex }} />
                          <span className="text-xs font-mono text-gray-600">{c.hex.toUpperCase()}</span>
                          <div className="flex-1 h-1 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-gray-400 rounded-full" style={{ width: `${c.percentage}%` }} />
                          </div>
                          <span className="text-xs text-gray-400 w-8 text-right">{c.percentage}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Tag className="w-4 h-4 text-orange-500" />
                      <h3 className="text-sm font-semibold text-gray-800">AI-Generated Tags</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {result.tags.map((tag) => (
                        <span key={tag} className="text-xs px-2.5 py-1.5 rounded-full bg-gradient-to-r from-indigo-50 to-purple-50 text-indigo-700 border border-indigo-100 font-medium">
                          #{tag}
                        </span>
                      ))}
                    </div>
                    <div className="mt-4 pt-3 border-t border-gray-100 grid grid-cols-2 gap-2">
                      <div className="text-center p-2 bg-gray-50 rounded-lg">
                        <p className="text-lg font-bold text-gray-800">{result.metadata.megapixels}</p>
                        <p className="text-xs text-gray-400">Megapixels</p>
                      </div>
                      <div className="text-center p-2 bg-gray-50 rounded-lg">
                        <p className="text-lg font-bold text-gray-800">{result.palette.length}</p>
                        <p className="text-xs text-gray-400">Key Colors</p>
                      </div>
                      <div className="text-center p-2 bg-gray-50 rounded-lg">
                        <p className="text-lg font-bold text-gray-800">{result.classification.length}</p>
                        <p className="text-xs text-gray-400">Scene Classes</p>
                      </div>
                      <div className="text-center p-2 bg-gray-50 rounded-lg">
                        <p className="text-lg font-bold text-gray-800">{result.tags.length}</p>
                        <p className="text-xs text-gray-400">Tags</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
