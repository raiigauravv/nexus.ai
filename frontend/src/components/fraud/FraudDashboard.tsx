"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  Activity,
  DollarSign,
  ShieldAlert,
  TrendingUp,
  Zap,
  Clock,
} from "lucide-react";
import { API_BASE } from "@/lib/api";

const BACKEND = API_BASE;

interface Transaction {
  transaction_id: string;
  amount: number;
  merchant_category: string;
  merchant_name: string;
  timestamp: string;
  velocity_1h: number;
  distance_from_home_km: number;
  cardholder_name: string;
}

interface Prediction {
  fraud_score: number;
  is_fraud: boolean;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  confidence: number;
  reasons: string[];
}

interface FraudEvent {
  transaction: Transaction;
  prediction: Prediction;
}

const RISK_STYLES = {
  LOW: {
    badge: "bg-emerald-100 text-emerald-700 border-emerald-200",
    bar: "bg-emerald-500",
    row: "border-emerald-100",
    icon: <CheckCircle className="w-4 h-4 text-emerald-500" />,
  },
  MEDIUM: {
    badge: "bg-amber-100 text-amber-700 border-amber-200",
    bar: "bg-amber-500",
    row: "border-amber-100",
    icon: <AlertTriangle className="w-4 h-4 text-amber-500" />,
  },
  HIGH: {
    badge: "bg-red-100 text-red-700 border-red-200",
    bar: "bg-red-500",
    row: "border-red-100",
    icon: <ShieldAlert className="w-4 h-4 text-red-500" />,
  },
};

const CATEGORY_EMOJI: Record<string, string> = {
  grocery: "🛒",
  gas_station: "⛽",
  restaurant: "🍽️",
  online_retail: "🛍️",
  travel: "✈️",
  electronics: "💻",
  pharmacy: "💊",
  entertainment: "🎬",
  atm: "🏧",
  luxury: "💎",
};

function formatTime(iso: string) {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "--:--";
  }
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    pct > 70 ? "bg-red-500" : pct > 40 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono font-semibold w-8 text-right">
        {pct}%
      </span>
    </div>
  );
}

export default function FraudDashboard() {
  const [events, setEvents] = useState<FraudEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stats, setStats] = useState({
    total: 0,
    flagged: 0,
    high_risk: 0,
    metrics: { f1: 0.0, precision: 0.0, recall: 0.0, auc_roc: 0.0 },
  });
  const esRef = useRef<EventSource | null>(null);
  const MAX_EVENTS = 50;

  const startStream = () => {
    if (esRef.current) return;
    setIsStreaming(true);
    const es = new EventSource(`${BACKEND}/fraud/stream?interval_ms=1800`);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const event: FraudEvent = JSON.parse(e.data);
        setEvents((prev) => {
          const next = [event, ...prev].slice(0, MAX_EVENTS);
          return next;
        });
        setStats((prev) => ({
          ...prev,
          total: prev.total + 1,
          flagged:
            prev.flagged + (event.prediction.is_fraud ? 1 : 0),
          high_risk:
            prev.high_risk +
            (event.prediction.risk_level === "HIGH" ? 1 : 0),
        }));
      } catch {
        /* ignore malformed */
      }
    };

    es.onerror = () => {
      es.close();
      esRef.current = null;
      setIsStreaming(false);
    };
  };

  const stopStream = () => {
    esRef.current?.close();
    esRef.current = null;
    setIsStreaming(false);
  };

  useEffect(() => {
    startStream();
    
    // Fetch real metrics from the holdout dataset
    fetch(`${BACKEND}/fraud/stats`)
      .then((res) => res.json())
      .then((data) => {
        if (data.metrics) {
          setStats((prev) => ({ ...prev, metrics: data.metrics }));
        }
      })
      .catch((err) => console.error("Could not fetch fraud metrics:", err));

    return () => stopStream();
  }, []);

  const fraudRate =
    stats.total > 0
      ? ((stats.flagged / stats.total) * 100).toFixed(1)
      : "0.0";

  const alerts = events.filter((e) => e.prediction.is_fraud).slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Activity className="w-5 h-5 text-indigo-500" />}
          label="Analyzed"
          value={stats.total.toLocaleString()}
          sub="total transactions"
          accent="indigo"
        />
        <StatCard
          icon={<ShieldAlert className="w-5 h-5 text-red-500" />}
          label="Flagged"
          value={stats.flagged.toString()}
          sub={`${fraudRate}% fraud rate`}
          accent="red"
        />
        <StatCard
          icon={<AlertTriangle className="w-5 h-5 text-amber-500" />}
          label="High Risk"
          value={stats.high_risk.toString()}
          sub="score > 70%"
          accent="amber"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5 text-emerald-500" />}
          label="F1 Score"
          value={stats.metrics.f1 > 0 ? `${(stats.metrics.f1 * 100).toFixed(1)}%` : "..."}
          sub={`AUC: ${stats.metrics.auc_roc.toFixed(2)}`}
          accent="emerald"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Transaction Feed */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-indigo-500" />
              <h3 className="font-semibold text-gray-800 text-sm">
                Live Transaction Feed
              </h3>
              {isStreaming && (
                <span className="flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-indigo-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
                </span>
              )}
            </div>
            <div className="flex gap-2">
              {!isStreaming ? (
                <button
                  onClick={startStream}
                  className="text-xs px-3 py-1.5 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition-colors"
                >
                  Start
                </button>
              ) : (
                <button
                  onClick={stopStream}
                  className="text-xs px-3 py-1.5 bg-gray-600 text-white rounded-full hover:bg-gray-700 transition-colors"
                >
                  Pause
                </button>
              )}
            </div>
          </div>

          <div className="overflow-y-auto max-h-[440px]">
            {events.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 text-gray-400">
                <Activity className="w-8 h-8 mb-2 opacity-40" />
                <p className="text-sm">Waiting for transactions...</p>
              </div>
            ) : (
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-50 border-b border-gray-100">
                  <tr>
                    <th className="text-left px-4 py-2.5 font-medium text-gray-500">TXN ID</th>
                    <th className="text-left px-4 py-2.5 font-medium text-gray-500">Merchant</th>
                    <th className="text-right px-4 py-2.5 font-medium text-gray-500">Amount</th>
                    <th className="text-left px-4 py-2.5 font-medium text-gray-500 w-32">Score</th>
                    <th className="text-center px-4 py-2.5 font-medium text-gray-500">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((e, i) => {
                    const style = RISK_STYLES[e.prediction.risk_level];
                    const emoji =
                      CATEGORY_EMOJI[e.transaction.merchant_category] || "💳";
                    return (
                      <tr
                        key={`${e.transaction.transaction_id}-${i}`}
                        className={`border-b ${style.row} hover:bg-gray-50 transition-colors ${i === 0 ? "animate-pulse-once" : ""}`}
                      >
                        <td className="px-4 py-2.5 font-mono text-gray-500">
                          #{e.transaction.transaction_id}
                        </td>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-1.5">
                            <span>{emoji}</span>
                            <span className="text-gray-700 truncate max-w-[120px]">
                              {e.transaction.merchant_name}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-2.5 text-right font-semibold text-gray-800">
                          ${e.transaction.amount.toFixed(2)}
                        </td>
                        <td className="px-4 py-2.5 w-32">
                          <ScoreBar score={e.prediction.fraud_score} />
                        </td>
                        <td className="px-4 py-2.5 text-center">
                          <span
                            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${style.badge}`}
                          >
                            {style.icon}
                            {e.prediction.risk_level}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Alert Panel */}
        <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-red-500" />
            <h3 className="font-semibold text-gray-800 text-sm">
              Fraud Alerts
            </h3>
            {alerts.length > 0 && (
              <span className="ml-auto text-xs bg-red-100 text-red-700 rounded-full px-2 py-0.5 font-medium border border-red-200">
                {alerts.length}
              </span>
            )}
          </div>

          <div className="overflow-y-auto max-h-[440px] divide-y divide-gray-50">
            {alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 text-gray-400">
                <CheckCircle className="w-8 h-8 mb-2 opacity-40" />
                <p className="text-sm">No fraud detected</p>
              </div>
            ) : (
              alerts.map((e, i) => (
                <div key={`alert-${e.transaction.transaction_id}-${i}`} className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <p className="text-xs font-semibold text-gray-800">
                        {CATEGORY_EMOJI[e.transaction.merchant_category]}{" "}
                        {e.transaction.merchant_name}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatTime(e.transaction.timestamp)}
                      </p>
                    </div>
                    <span className="text-sm font-bold text-red-600">
                      ${e.transaction.amount.toFixed(2)}
                    </span>
                  </div>

                  <div className="mb-2">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Fraud Score</span>
                      <span className="font-semibold text-red-600">
                        {(e.prediction.fraud_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red-500 rounded-full"
                        style={{
                          width: `${e.prediction.fraud_score * 100}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="space-y-1">
                    {e.prediction.reasons.slice(0, 2).map((r, ri) => (
                      <p
                        key={ri}
                        className="text-xs text-gray-500 flex items-start gap-1"
                      >
                        <span className="text-red-400 mt-0.5">•</span>
                        {r}
                      </p>
                    ))}
                  </div>

                  <p className="text-xs text-gray-400 mt-2">
                    Cardholder: {e.transaction.cardholder_name} ·{" "}
                    #{e.transaction.transaction_id}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  accent: string;
}) {
  const accentMap: Record<string, string> = {
    indigo: "border-indigo-100 bg-indigo-50/40",
    red: "border-red-100 bg-red-50/40",
    amber: "border-amber-100 bg-amber-50/40",
    emerald: "border-emerald-100 bg-emerald-50/40",
  };

  return (
    <div
      className={`rounded-xl border p-4 shadow-sm ${accentMap[accent] || "border-gray-100 bg-white"}`}
    >
      <div className="flex items-center gap-2 mb-2">{icon} <span className="text-xs font-medium text-gray-500">{label}</span></div>
      <p className="text-2xl font-bold text-gray-800">{value}</p>
      <p className="text-xs text-gray-400 mt-0.5">{sub}</p>
    </div>
  );
}
