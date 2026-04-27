"use client";

import { useState, useRef } from "react";
import {
  BarChart2,
  Send,
  Smile,
  Meh,
  Frown,
  Sparkles,
  BookOpen,
  Star,
  RefreshCw,
  ChevronRight,
  Activity,
  Eye,
} from "lucide-react";
import { API_BASE } from "@/lib/api";

const BACKEND = API_BASE;

interface SentimentOverall {
  label: "positive" | "neutral" | "negative";
  score: number;
  emoji: string;
  confidence: number;
}

interface AspectResult {
  aspect: string;
  sentiment: "positive" | "neutral" | "negative";
  score: number;
  matched_keyword: string;
  excerpt: string;
}

interface EmotionScores {
  joy: number;
  trust: number;
  anticipation: number;
  sadness: number;
  anger: number;
  fear: number;
  surprise: number;
  disgust: number;
}

interface Scores {
  vader_compound: number;
  textblob_polarity: number;
  textblob_subjectivity: number;
  ensemble: number;
}

interface Metadata {
  word_count: number;
  sentence_count: number;
  subjectivity: number;
  readability_score: number;
  readability_label: string;
}

interface AnalysisResult {
  text: string;
  overall: SentimentOverall;
  model_info: Scores;
  metadata: Metadata;
  aspects: AspectResult[];
  emotions: EmotionScores;
}

interface Sample {
  id: string;
  product: string;
  category: string;
  author: string;
  text: string;
  stars: number;
}

const SENTIMENT_CONFIG = {
  positive: {
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    text: "text-emerald-700",
    bar: "bg-emerald-500",
    icon: <Smile className="w-5 h-5" />,
    label: "Positive",
  },
  neutral: {
    bg: "bg-amber-50",
    border: "border-amber-200",
    text: "text-amber-700",
    bar: "bg-amber-500",
    icon: <Meh className="w-5 h-5" />,
    label: "Neutral",
  },
  negative: {
    bg: "bg-red-50",
    border: "border-red-200",
    text: "text-red-700",
    bar: "bg-red-500",
    icon: <Frown className="w-5 h-5" />,
    label: "Negative",
  },
};

const EMOTION_CONFIG: Record<string, { color: string; emoji: string }> = {
  joy:          { color: "bg-yellow-400", emoji: "😄" },
  trust:        { color: "bg-emerald-500", emoji: "🤝" },
  anticipation: { color: "bg-orange-400", emoji: "⚡" },
  sadness:      { color: "bg-blue-400", emoji: "😢" },
  anger:        { color: "bg-red-500", emoji: "😠" },
  fear:         { color: "bg-purple-400", emoji: "😨" },
  surprise:     { color: "bg-pink-400", emoji: "😲" },
  disgust:      { color: "bg-lime-600", emoji: "🤢" },
};

function StarDisplay({ count }: { count: number }) {
  return (
    <div className="flex gap-0.5">
      {Array.from({ length: 5 }).map((_, i) => (
        <Star
          key={i}
          className={`w-3.5 h-3.5 ${i < count ? "fill-amber-400 text-amber-400" : "text-gray-200"}`}
        />
      ))}
    </div>
  );
}

function GaugeMeter({ score }: { score: number }) {
  // score in [-1, 1]; map to [0, 180] degrees
  const pct = (score + 1) / 2; // 0–1
  const angle = pct * 180; // 0–180 deg
  const color = score > 0.05 ? "#10b981" : score < -0.05 ? "#ef4444" : "#f59e0b";

  const cx = 80, cy = 80, r = 60;
  const startAngle = 180; // left
  const endAngle = startAngle - angle; // rotate CCW
  const toRad = (d: number) => (d * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startAngle));
  const y1 = cy + r * Math.sin(toRad(startAngle));
  const x2 = cx + r * Math.cos(toRad(endAngle));
  const y2 = cy + r * Math.sin(toRad(endAngle));
  const largeArc = angle > 180 ? 1 : 0;

  return (
    <svg viewBox="0 0 160 90" className="w-40 h-24">
      {/* Background arc */}
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none"
        stroke="#e5e7eb"
        strokeWidth="12"
        strokeLinecap="round"
      />
      {/* Colored arc */}
      {angle > 0 && (
        <path
          d={`M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 0 ${x2} ${y2}`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
        />
      )}
      {/* Center labels */}
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize="18" fontWeight="bold" fill={color}>
        {score >= 0 ? "+" : ""}{score.toFixed(2)}
      </text>
      <text x={cx} y={cy + 14} textAnchor="middle" fontSize="10" fill="#9ca3af">
        Ensemble Score
      </text>
      {/* Side labels */}
      <text x="8" y={cy + 4} fontSize="10" fill="#ef4444">NEG</text>
      <text x={cx} y="18" textAnchor="middle" fontSize="10" fill="#f59e0b">NEU</text>
      <text x="128" y={cy + 4} fontSize="10" fill="#10b981">POS</text>
    </svg>
  );
}

function AspectBadge({ aspect }: { aspect: AspectResult }) {
  const cfg = SENTIMENT_CONFIG[aspect.sentiment];
  return (
    <div className={`rounded-xl border p-3 ${cfg.bg} ${cfg.border}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-semibold text-gray-700">{aspect.aspect}</span>
        <span className={`text-xs font-bold ${cfg.text}`}>
          {aspect.sentiment === "positive" ? "+" : aspect.sentiment === "negative" ? "−" : "·"}
          {Math.abs(aspect.score).toFixed(2)}
        </span>
      </div>
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className={`text-xs ${cfg.text}`}>{cfg.icon}</span>
        <span className={`text-xs font-medium capitalize ${cfg.text}`}>{aspect.sentiment}</span>
        <span className="text-xs text-gray-400 ml-auto">#{aspect.matched_keyword}</span>
      </div>
      <div className="h-1 bg-white/60 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${cfg.bar}`}
          style={{ width: `${Math.abs(aspect.score) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function SentimentDashboard() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [activeTab, setActiveTab] = useState<"analyze" | "samples">("analyze");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const DEMO_TEXTS = [
    "This product is absolutely incredible! Best purchase I've made all year. The quality is outstanding.",
    "Completely disappointed. Broke after one week. Customer service was useless and rude.",
    "It's okay. Nothing special. Does what it says but nothing more. Average quality for the price.",
  ];

  const analyze = async (text: string) => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const resp = await fetch(`${BACKEND}/sentiment/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadSamples = async () => {
    setLoadingSamples(true);
    try {
      const resp = await fetch(`${BACKEND}/sentiment/samples`);
      const data = await resp.json();
      setSamples(data.samples || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingSamples(false);
    }
  };

  const analyzeSample = async (sample: Sample) => {
    setActiveTab("analyze");
    setInputText(sample.text);
    await analyze(sample.text);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    analyze(inputText);
  };

  const cfg = result ? SENTIMENT_CONFIG[result.overall.label] : null;

  return (
    <div className="space-y-6">
      {/* Tabs */}
      <div className="flex items-center gap-2">
        <div className="flex bg-gray-100 rounded-xl p-1 gap-1">
          <button
            onClick={() => setActiveTab("analyze")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === "analyze"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <Activity className="w-4 h-4" />
            Live Analyzer
          </button>
          <button
            onClick={() => { setActiveTab("samples"); if (!samples.length) loadSamples(); }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === "samples"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Sample Reviews
          </button>
        </div>
      </div>

      {activeTab === "analyze" && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Input Panel */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-500" />
                Analyze Text
              </h3>
              <form onSubmit={handleSubmit} className="space-y-3">
                <textarea
                  ref={textareaRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  rows={6}
                  placeholder="Paste any text, review, tweet, or feedback here..."
                  className="w-full text-sm rounded-xl border border-gray-200 bg-gray-50 p-3 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 transition-all placeholder-gray-400 text-gray-800"
                />
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">{inputText.length}/5000</span>
                  <button
                    type="submit"
                    disabled={loading || !inputText.trim()}
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                  >
                    {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              </form>
            </div>

            {/* Quick Demo */}
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Quick Demo</p>
              <div className="space-y-2">
                {DEMO_TEXTS.map((t, i) => (
                  <button
                    key={i}
                    onClick={() => { setInputText(t); analyze(t); }}
                    className="w-full text-left text-xs text-gray-600 p-2.5 rounded-lg bg-gray-50 hover:bg-indigo-50 hover:text-indigo-700 border border-gray-100 hover:border-indigo-200 transition-all leading-relaxed flex items-start gap-2"
                  >
                    <ChevronRight className="w-3 h-3 mt-0.5 flex-shrink-0 text-gray-400" />
                    {t.length > 80 ? t.slice(0, 80) + "…" : t}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-3 space-y-4">
            {!result && !loading && (
              <div className="bg-white rounded-xl border border-dashed border-gray-200 h-64 flex flex-col items-center justify-center text-gray-400">
                <BarChart2 className="w-10 h-10 opacity-30 mb-3" />
                <p className="text-sm">Enter text and click Analyze</p>
                <p className="text-xs mt-1 opacity-60">or pick a demo text on the left</p>
              </div>
            )}

            {loading && (
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm h-64 flex flex-col items-center justify-center gap-3">
                <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin" />
                <p className="text-sm text-gray-500">Running NLP pipeline...</p>
              </div>
            )}

            {result && !loading && cfg && (
              <>
                {/* Overall Sentiment Card */}
                <div className={`rounded-xl border p-5 ${cfg.bg} ${cfg.border}`}>
                  <div className="flex items-start gap-4">
                    {/* Gauge */}
                    <div className="flex-shrink-0">
                      <GaugeMeter score={result.overall.score} />
                    </div>
                    <div className="flex-1">
                      <div className={`flex items-center gap-2 mb-1 ${cfg.text}`}>
                        {cfg.icon}
                        <span className="text-lg font-bold">{cfg.label} Sentiment</span>
                        <span className="text-2xl">{result.overall.emoji}</span>
                      </div>
                      <p className="text-sm text-gray-600 mb-3">
                        Confidence: <span className="font-semibold">{Math.round(result.overall.confidence * 100)}%</span>
                      </p>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <MetaChip label="Words" value={result.metadata.word_count.toString()} />
                        <MetaChip label="Sentences" value={result.metadata.sentence_count.toString()} />
                        <MetaChip label="Subjectivity" value={`${Math.round(result.metadata.subjectivity * 100)}%`} />
                        <MetaChip label="Readability" value={result.metadata.readability_label} />
                        <MetaChip label="VADER" value={result.model_info?.vader_compound?.toFixed(3) || "N/A"} />
                        <MetaChip label="TextBlob" value={result.model_info?.textblob_polarity?.toFixed(3) || "N/A"} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Emotion Wheel */}
                {result.emotions && (
                  <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1">
                      <Eye className="w-3.5 h-3.5" /> Emotion Distribution
                    </p>
                    <div className="space-y-2">
                      {Object.entries(result.emotions)
                        .sort(([, a], [, b]) => b - a)
                        .map(([emotion, value]) => {
                          const ec = EMOTION_CONFIG[emotion];
                          const pct = Math.round(value * 100);
                          return (
                            <div key={emotion} className="flex items-center gap-2">
                              <span className="text-sm w-6">{ec?.emoji}</span>
                              <span className="text-xs text-gray-600 capitalize w-24">{emotion}</span>
                              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-700 ${ec?.color || "bg-gray-400"}`}
                                  style={{ width: `${Math.min(100, pct * 5)}%` }}
                                />
                              </div>
                              <span className="text-xs font-mono text-gray-500 w-8 text-right">{value.toFixed(3)}</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}

                {/* Aspect Breakdown */}
                {result.aspects.length > 0 && (
                  <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                      Aspect-Level Sentiment
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {result.aspects.map((aspect) => (
                        <AspectBadge key={aspect.aspect} aspect={aspect} />
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Samples Tab */}
      {activeTab === "samples" && (
        <div>
          {loadingSamples ? (
            <div className="flex items-center justify-center h-40">
              <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {samples.map((sample) => (
                <div
                  key={sample.id}
                  className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 hover:shadow-md hover:border-indigo-100 transition-all cursor-pointer group"
                  onClick={() => analyzeSample(sample)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <p className="font-semibold text-gray-800 text-sm group-hover:text-indigo-700 transition-colors">
                        {sample.product}
                      </p>
                      <p className="text-xs text-gray-400 mt-0.5">{sample.category} · {sample.author}</p>
                    </div>
                    <StarDisplay count={sample.stars} />
                  </div>
                  <p className="text-xs text-gray-600 leading-relaxed line-clamp-3">{sample.text}</p>
                  <div className="mt-3 flex items-center text-xs text-indigo-500 font-medium gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Sparkles className="w-3 h-3" /> Click to analyze
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetaChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white/70 rounded-lg px-2 py-1.5 text-center border border-white/80">
      <p className="text-gray-400 text-xs">{label}</p>
      <p className="font-semibold text-gray-700">{value}</p>
    </div>
  );
}
