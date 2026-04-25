"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Cpu,
  Star,
  TrendingUp,
  RefreshCw,
  ChevronRight,
  Sparkles,
  Users,
  Tag,
  Flame,
} from "lucide-react";

const BACKEND = "http://localhost:8000/api/v1";

interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  rating: number;
  tags: string[];
  recommendation_score?: number;
  match_reason?: string;
  trending_score?: number;
  interaction_count?: number;
  similarity_score?: number;
  // Cross-module fields
  sentiment_health?: number;
  sentiment_health_label?: string;
  fraud_flag?: string | null;
}

interface User {
  id: string;
  name: string;
  avatar: string;
  persona: string;
}

interface RecommendationResult {
  user: User;
  recommendations: Product[];
  algorithm: string;
}

const CATEGORY_COLORS: Record<string, string> = {
  Electronics:   "bg-blue-100 text-blue-700",
  Books:         "bg-amber-100 text-amber-700",
  "Clothing":    "bg-purple-100 text-purple-700",
  "Home & Kitchen": "bg-emerald-100 text-emerald-700",
  Sports:        "bg-orange-100 text-orange-700",
  Gaming:        "bg-red-100 text-red-700",
  Beauty:        "bg-pink-100 text-pink-700",
  Automotive:    "bg-slate-100 text-slate-700",
};

const CATEGORY_EMOJI: Record<string, string> = {
  Electronics: "💻",
  Books: "📚",
  Clothing: "👗",
  "Home & Kitchen": "🏠",
  Sports: "🏃",
  Gaming: "🎮",
  Beauty: "✨",
  Automotive: "🚗",
};

const PERSONA_LABELS: Record<string, string> = {
  tech_professional: "Tech Professional",
  fitness_enthusiast: "Fitness Enthusiast",
  gamer: "Gamer",
  bookworm: "Bookworm",
  home_chef: "Home Chef",
  fashionista: "Fashionista",
  outdoor_adventurer: "Outdoor Adventurer",
  beauty_enthusiast: "Beauty Enthusiast",
};

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-1">
      <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
      <span className="text-xs font-semibold text-gray-700">{rating.toFixed(1)}</span>
    </div>
  );
}

function SentimentBadge({ label }: { label: string }) {
  if (!label) return null;
  const color = label.includes("Loved")
    ? "text-emerald-600 bg-emerald-50 border-emerald-200"
    : label.includes("Poor")
    ? "text-red-600 bg-red-50 border-red-200"
    : "text-amber-600 bg-amber-50 border-amber-200";
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${color}`}>
      {label}
    </span>
  );
}

function ScorePill({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct > 70 ? "text-emerald-600 bg-emerald-50 border-emerald-200"
    : pct > 40 ? "text-indigo-600 bg-indigo-50 border-indigo-200"
    : "text-gray-500 bg-gray-50 border-gray-200";
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${color}`}>
      {pct}% match
    </span>
  );
}

function ProductCard({ product, showScore = true, showReason = true }: {
  product: Product;
  showScore?: boolean;
  showReason?: boolean;
}) {
  const emoji = CATEGORY_EMOJI[product.category] || "📦";
  const colorClass = CATEGORY_COLORS[product.category] || "bg-gray-100 text-gray-700";

  return (
    <div className="bg-white rounded-xl border border-gray-100 p-4 hover:shadow-md hover:border-indigo-100 transition-all duration-200 group cursor-pointer">
      {/* Category & Score Row */}
      <div className="flex items-center justify-between mb-3">
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${colorClass}`}>
          {emoji} {product.category}
        </span>
        {showScore && product.recommendation_score !== undefined && (
          <ScorePill score={product.recommendation_score} />
        )}
        {product.trending_score !== undefined && (
          <div className="flex items-center gap-1 text-xs text-orange-600 font-semibold">
            <Flame className="w-3 h-3" />
            {product.interaction_count} interactions
          </div>
        )}
        {product.similarity_score !== undefined && (
          <span className="text-xs text-indigo-600 font-semibold bg-indigo-50 px-2 py-0.5 rounded-full border border-indigo-100">
            {Math.round(product.similarity_score * 100)}% similar
          </span>
        )}
      </div>

      {/* Product Name */}
      <h4 className="text-sm font-semibold text-gray-800 mb-1 group-hover:text-indigo-700 transition-colors leading-snug">
        {product.name}
      </h4>

      {/* Cross-module signals */}
      {product.sentiment_health_label && (
        <div className="flex items-center gap-2 mt-1.5 mb-1">
          <SentimentBadge label={product.sentiment_health_label} />
          {product.fraud_flag && (
            <span className="text-xs text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full font-medium truncate max-w-[160px]" title={product.fraud_flag}>
              {product.fraud_flag}
            </span>
          )}
        </div>
      )}

      {/* Price + Rating */}
      <div className="flex items-center justify-between mt-2">
        <span className="text-lg font-bold text-gray-900">${product.price.toFixed(2)}</span>
        <StarRating rating={product.rating} />
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mt-3">
        {product.tags.slice(0, 3).map((tag) => (
          <span key={tag} className="text-xs bg-gray-100 text-gray-500 rounded px-1.5 py-0.5">
            #{tag}
          </span>
        ))}
      </div>

      {/* Match reason */}
      {showReason && product.match_reason && (
        <p className="text-xs text-indigo-500 mt-3 flex items-center gap-1 font-medium">
          <Sparkles className="w-3 h-3" />
          {product.match_reason}
        </p>
      )}
    </div>
  );
}

import { useAuth } from "@/context/AuthContext";

export default function RecommendationDashboard() {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [trending, setTrending] = useState<Product[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"for-you" | "trending">("for-you");
  const { user: authUser } = useAuth();

  // Load users + trending on mount
  useEffect(() => {
    fetch(`${BACKEND}/recommend/users`)
      .then((r) => r.json())
      .then((d) => {
        let fetchedUsers = d.users || [];
        setUsers(fetchedUsers);
      })
      .catch(console.error);

    fetch(`${BACKEND}/recommend/trending?top_n=8`)
      .then((r) => r.json())
      .then((d) => setTrending(d.trending || []))
      .catch(console.error);
  }, []);

  // Update selectedUser if authUser changes
  useEffect(() => {
    if (authUser) {
      setSelectedUser({
        id: authUser.id,
        name: authUser.name,
        avatar: "👤",
        persona: authUser.persona,
      });
    } else if (users.length > 0 && !selectedUser) {
      setSelectedUser(users[0]);
    }
  }, [authUser, users, selectedUser]);

  // Fetch recommendations when user changes
  const fetchRecommendations = useCallback(async (user: User) => {
    setLoading(true);
    try {
      const resp = await fetch(`${BACKEND}/recommend/for/${user.id}?top_n=6`);
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedUser) fetchRecommendations(selectedUser);
  }, [selectedUser, fetchRecommendations]);

  const handleUserSelect = (user: User) => {
    setSelectedUser(user);
    setActiveTab("for-you");
  };

  return (
    <div className="space-y-6">

      {/* User Selector */}
      <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
        <div className="flex items-center gap-2 mb-4">
          <Users className="w-4 h-4 text-indigo-500" />
          <h3 className="font-semibold text-gray-800 text-sm">Select User Profile</h3>
          <span className="ml-auto text-xs text-gray-400">{users.length} synthetic users</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {users.map((user) => (
            <button
              key={user.id}
              onClick={() => handleUserSelect(user)}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-sm transition-all duration-200 ${
                selectedUser?.id === user.id
                  ? "bg-indigo-600 text-white border-indigo-600 shadow-sm shadow-indigo-200"
                  : "bg-white border-gray-200 text-gray-700 hover:border-indigo-300 hover:bg-indigo-50/50"
              }`}
            >
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                selectedUser?.id === user.id ? "bg-indigo-500 text-white" : "bg-gray-100 text-gray-600"
              }`}>
                {user.avatar}
              </div>
              <div className="text-left">
                <p className="font-medium leading-none">{user.name}</p>
                <p className={`text-xs mt-0.5 ${selectedUser?.id === user.id ? "text-indigo-200" : "text-gray-400"}`}>
                  {PERSONA_LABELS[user.persona] || user.persona}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Row */}
      <div className="flex items-center gap-4">
        <div className="flex bg-gray-100 rounded-xl p-1 gap-1">
          <button
            onClick={() => setActiveTab("for-you")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === "for-you"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            For You
          </button>
          <button
            onClick={() => setActiveTab("trending")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === "trending"
                ? "bg-white text-orange-600 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <Flame className="w-4 h-4" />
            Trending
          </button>
        </div>

        {activeTab === "for-you" && result && (
          <div className="flex items-center gap-2 text-xs text-gray-400 ml-auto">
            <Cpu className="w-3.5 h-3.5" />
            {result.algorithm}
            <button
              onClick={() => selectedUser && fetchRecommendations(selectedUser)}
              className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 text-gray-400" />
            </button>
          </div>
        )}
      </div>

      {/* For You Panel */}
      {activeTab === "for-you" && (
        <div>
          {loading ? (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="bg-gray-100 rounded-xl h-48 animate-pulse" />
              ))}
            </div>
          ) : result ? (
            <div>
              {/* User context banner */}
              <div className="flex items-center gap-3 mb-4 p-3 bg-indigo-50 rounded-xl border border-indigo-100">
                <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                  {result.user?.avatar}
                </div>
                <div>
                  <p className="text-sm font-semibold text-indigo-800">
                    Showing recommendations for <span className="font-bold">{result.user?.name}</span>
                  </p>
                  <p className="text-xs text-indigo-500">
                    Persona: {PERSONA_LABELS[result.user?.persona] || result.user?.persona} · {result.recommendations.length} items curated
                  </p>
                </div>
                <div className="ml-auto flex items-center gap-1 text-xs text-indigo-400">
                  <TrendingUp className="w-3.5 h-3.5" />
                  Personalized
                </div>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                {result.recommendations.map((product) => (
                  <ProductCard
                    key={product.id}
                    product={product}
                    showScore
                    showReason
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-48 text-gray-400">
              <Cpu className="w-10 h-10 opacity-30 mb-3" />
              <p className="text-sm">Select a user to see recommendations</p>
            </div>
          )}
        </div>
      )}

      {/* Trending Panel */}
      {activeTab === "trending" && (
        <div>
          <div className="flex items-center gap-2 mb-4 text-xs text-gray-500">
            <Flame className="w-3.5 h-3.5 text-orange-500" />
            <span>Trending across all users — ranked by Bayesian-weighted popularity</span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {trending.map((product, i) => (
              <div key={product.id} className="relative">
                {i < 3 && (
                  <div className={`absolute -top-2 -left-2 z-10 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white shadow
                    ${i === 0 ? "bg-amber-500" : i === 1 ? "bg-gray-400" : "bg-orange-500"}`}>
                    {i + 1}
                  </div>
                )}
                <ProductCard product={product} showScore={false} showReason={false} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
